"""Interactive matplotlib GUI for annotating the 2D embedding plane.

The GUI shows the 2D histogram of the (reduced) embedding and lets the user
partition it into classes with a Gaussian-mixture model. Three ways to drive
the partition are offered:

* **Fit K classes** — automatic GMM for the number of classes on the slider.
* **Auto-K (BIC)** — automatic GMM whose number of classes minimises the BIC.
* **Fit seeds** — GMM initialised from the placed seeds, then refined by EM.
* **Apply seeds (manual)** — no fitting: each seed becomes a fixed spherical
  component using its placed position (mean) and radius (variance), and points
  are assigned to the nearest such component.

In seed mode, left-click to add a seed, drag a seed to move it, scroll over a
seed to grow/shrink its radius, and right-click a seed to delete it. The
covariance model used by the (non-manual) GMM fits is selectable.

The partition is exclusive and exhaustive, so every particle always belongs to
exactly one class. On start-up all particles share a single class, preserving
that invariant even before the user acts.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse as EllipsePatch
from matplotlib.widgets import Button, RadioButtons, Slider

from .partition import (
    ELLIPSE_N_STD,
    GmmPartitioner,
    ManualSphericalPartitioner,
    Partitioner,
    fit_gmm_bic,
)

#: Covariance models offered for the GMM fits.
_COVARIANCE_TYPES = ('full', 'tied', 'diag', 'spherical')

logger = logging.getLogger(__name__)

#: Cap on the number of points drawn in the scatter overlay, for responsiveness.
_MAX_SCATTER = 50_000

#: Pixel radius within which a click is considered to hit an existing seed.
_SEED_PICK_RADIUS_PX = 10.0


class AnnotationApp:
    """Interactive annotation of a 2D point cloud into exclusive classes."""

    def __init__(
        self,
        points: np.ndarray,
        bins: int = 200,
        initial_classes: int = 3,
        max_classes: int = 20,
        covariance_type: str = 'full',
        cmap: str = 'tab20',
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must be an (N, 2) array')
        if covariance_type not in _COVARIANCE_TYPES:
            raise ValueError(
                f'covariance_type must be one of {_COVARIANCE_TYPES}'
            )

        self.points = points
        self.n = points.shape[0]
        self.bins = bins
        self.initial_classes = int(np.clip(initial_classes, 1, max_classes))
        self.max_classes = max_classes
        self.covariance_type = covariance_type
        self.cmap = plt.get_cmap(cmap)

        # Default seed radius (sigma) scaled to the data extent.
        self._default_sigma = 0.04 * float(np.max(np.ptp(points, axis=0)) or 1.0)

        # Every particle starts in a single class -> partition invariant holds.
        self.labels = np.zeros(self.n, dtype=np.int64)
        self.partitioner = None
        # Each seed is (x, y, sigma): a mean and a spherical radius.
        self.seeds: List[Tuple[float, float, float]] = []
        self.seed_mode = False
        self.saved = False
        self._drag_index: Optional[int] = None

        # Subsample the scatter overlay for large datasets (labels stay full).
        if self.n > _MAX_SCATTER:
            rng = np.random.default_rng(0)
            self._disp = rng.choice(self.n, _MAX_SCATTER, replace=False)
        else:
            self._disp = np.arange(self.n)

        self._scatter = None
        self._seed_artist = None
        self._seed_circles: List[Circle] = []
        self._ellipses: List[EllipsePatch] = []

        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        self.fig = plt.figure(figsize=(11, 7))
        try:
            self.fig.canvas.manager.set_window_title('sinovar annotate')
        except AttributeError:
            pass

        gs = GridSpec(
            1, 2, width_ratios=[4, 1], figure=self.fig,
            left=0.07, right=0.98, top=0.94, bottom=0.08, wspace=0.2,
        )
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.set_title('sinovarEmbedding (first 2 components)')
        self.ax.set_xlabel('component 1')
        self.ax.set_ylabel('component 2')

        # 2D histogram as the density background.
        self.ax.hist2d(
            self.points[:, 0], self.points[:, 1],
            bins=self.bins, cmap='Greys', cmin=1,
        )

        self._build_controls()
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._redraw()

    def _build_controls(self) -> None:
        # Widget axes are placed in figure coordinates in the right-hand column.
        left, width = 0.80, 0.17

        def button(y: float, label: str, callback) -> Button:
            axis = self.fig.add_axes([left, y, width, 0.048])
            widget = Button(axis, label)
            widget.on_clicked(callback)
            return widget

        # --- automatic GMM controls ---
        self.fig.text(left, 0.955, 'Covariance type', fontsize=9, weight='bold')
        self.ax_cov = self.fig.add_axes([left, 0.85, width, 0.10])
        self.radio_cov = RadioButtons(
            self.ax_cov, _COVARIANCE_TYPES,
            active=_COVARIANCE_TYPES.index(self.covariance_type),
        )
        self.radio_cov.on_clicked(self._on_covariance)

        self.ax_slider = self.fig.add_axes([left, 0.812, width, 0.025])
        self.slider_k = Slider(
            self.ax_slider, 'K', 1, self.max_classes,
            valinit=self.initial_classes, valstep=1,
        )
        self.btn_fit_k = button(0.754, 'Fit K classes', self._on_fit_k)
        self.btn_auto_k = button(0.696, 'Auto-K (BIC)', self._on_auto_k)

        # --- manual / seed controls ---
        self.btn_seed = button(0.616, 'Seed mode: OFF', self._on_toggle_seed)
        self.btn_fit_seeds = button(0.558, 'Fit seeds (EM)', self._on_fit_seeds)
        self.btn_apply_manual = button(
            0.500, 'Apply seeds (manual)', self._on_apply_manual
        )
        self.btn_clear_seeds = button(0.442, 'Clear seeds', self._on_clear_seeds)

        # --- global controls ---
        self.btn_clear_annotation = button(
            0.362, 'Clear annotation', self._on_clear_annotation
        )
        self.btn_save = button(0.304, 'Save & close', self._on_save)

        self.status_text = self.fig.text(
            left, 0.27, '', fontsize=9, va='top', wrap=True,
        )
        self.counts_text = self.fig.text(
            left, 0.17, '', fontsize=8, va='top', family='monospace',
        )

    # -------------------------------------------------------------- callbacks
    def _on_covariance(self, label: str) -> None:
        self.covariance_type = label
        self._status(f"Covariance type: '{label}'")

    def _on_fit_k(self, _event) -> None:
        k = int(self.slider_k.val)
        self._apply(
            GmmPartitioner(n_components=k, covariance_type=self.covariance_type),
            f"Fitted GMM, K={k} ('{self.covariance_type}')",
        )

    def _on_auto_k(self, _event) -> None:
        self._status('Searching K by BIC...')
        self.fig.canvas.draw_idle()
        best = fit_gmm_bic(
            self.points, range(1, self.max_classes + 1),
            covariance_type=self.covariance_type,
        )
        self.partitioner = best
        self.labels = best.labels_
        self.slider_k.set_val(best.n_components)
        self._status(f'Auto-K (BIC): K={best.n_components}')
        self._redraw()

    def _on_toggle_seed(self, _event) -> None:
        self.seed_mode = not self.seed_mode
        self.btn_seed.label.set_text(
            f'Seed mode: {"ON" if self.seed_mode else "OFF"}'
        )
        self._status(
            'Left-click to add, drag to move, right-click to delete seeds'
            if self.seed_mode else 'Seed mode off'
        )
        self.fig.canvas.draw_idle()

    def _on_fit_seeds(self, _event) -> None:
        if not self.seeds:
            self._status('Place at least one seed first')
            self.fig.canvas.draw_idle()
            return
        means = np.asarray(self.seeds, dtype=np.float64)[:, :2]
        self._apply(
            GmmPartitioner(
                n_components=len(self.seeds),
                means_init=means,
                covariance_type=self.covariance_type,
            ),
            f"Fitted GMM from {len(self.seeds)} seed(s) ('{self.covariance_type}')",
        )

    def _on_apply_manual(self, _event) -> None:
        if not self.seeds:
            self._status('Place at least one seed first')
            self.fig.canvas.draw_idle()
            return
        seeds = np.asarray(self.seeds, dtype=np.float64)
        means = seeds[:, :2]
        variances = seeds[:, 2] ** 2
        self._apply(
            ManualSphericalPartitioner(means, variances),
            f'Applied {len(self.seeds)} manual spherical component(s)',
        )

    def _on_clear_seeds(self, _event) -> None:
        self.seeds = []
        self._draw_seeds()
        self._status('Seeds cleared')

    def _on_clear_annotation(self, _event) -> None:
        # Drop the fitted model and return every particle to a single class,
        # which removes the ellipses and resets the point colouring.
        self.partitioner = None
        self.labels = np.zeros(self.n, dtype=np.int64)
        self._status('Annotation cleared')
        self._redraw()

    def _on_save(self, _event) -> None:
        self.saved = True
        plt.close(self.fig)

    def _seed_index_at(self, event) -> Optional[int]:
        """Return the index of the seed under the cursor, or ``None``.

        Hit-testing is done in pixel space so the pick radius is independent
        of the current zoom level.
        """
        if not self.seeds:
            return None
        means = np.asarray(self.seeds)[:, :2]
        display = self.ax.transData.transform(means)
        distances = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= _SEED_PICK_RADIUS_PX:
            return nearest
        return None

    def _on_press(self, event) -> None:
        if not self.seed_mode or event.inaxes is not self.ax:
            return
        # Ignore clicks while a navigation tool (zoom/pan) is active.
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return

        index = self._seed_index_at(event)
        if event.button == 3:  # right-click deletes the seed under the cursor
            if index is not None:
                del self.seeds[index]
                self._status(f'Seed deleted ({len(self.seeds)} remaining)')
                self._draw_seeds()
            return

        if event.button == 1:
            if index is not None:  # start dragging the existing seed
                self._drag_index = index
                self._status('Dragging seed (release to drop)')
            else:  # add a new seed on empty space
                self.seeds.append(
                    (event.xdata, event.ydata, self._default_sigma)
                )
                self._status(f'{len(self.seeds)} seed(s) placed')
                self._draw_seeds()

    def _on_motion(self, event) -> None:
        if self._drag_index is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        sigma = self.seeds[self._drag_index][2]  # preserve the radius
        self.seeds[self._drag_index] = (event.xdata, event.ydata, sigma)
        self._draw_seeds()

    def _on_release(self, _event) -> None:
        if self._drag_index is not None:
            self._drag_index = None
            self._status(f'{len(self.seeds)} seed(s) placed')

    def _on_scroll(self, event) -> None:
        if not self.seed_mode or event.inaxes is not self.ax:
            return
        index = self._seed_index_at(event)
        if index is None:
            return
        # Scrolling grows/shrinks the radius (sigma) of the hovered seed.
        factor = 1.2 if event.step > 0 else 1.0 / 1.2
        x, y, sigma = self.seeds[index]
        sigma *= factor
        self.seeds[index] = (x, y, sigma)
        self._status(f'Seed radius: sigma={sigma:.3g}')
        self._draw_seeds()

    # ----------------------------------------------------------------- helpers
    def _apply(self, partitioner: Partitioner, message: str) -> None:
        self.labels = partitioner.fit_predict(self.points)
        self.partitioner = partitioner
        self._status(message)
        self._redraw()

    def _status(self, message: str) -> None:
        self.status_text.set_text(message)

    def _draw_seeds(self) -> None:
        if self._seed_artist is not None:
            self._seed_artist.remove()
            self._seed_artist = None
        for circle in self._seed_circles:
            circle.remove()
        self._seed_circles = []

        if self.seeds:
            seeds = np.asarray(self.seeds)
            self._seed_artist = self.ax.scatter(
                seeds[:, 0], seeds[:, 1],
                marker='x', c='red', s=90, linewidths=2, zorder=6,
            )
            # Draw each seed's radius as the sigma circle it will contribute.
            for x, y, sigma in self.seeds:
                circle = Circle(
                    (x, y), ELLIPSE_N_STD * sigma,
                    fill=False, edgecolor='red', ls='--', lw=1.2,
                    alpha=0.8, zorder=5,
                )
                self.ax.add_patch(circle)
                self._seed_circles.append(circle)
        self.fig.canvas.draw_idle()

    def _redraw(self) -> None:
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        for ellipse in self._ellipses:
            ellipse.remove()
        self._ellipses = []

        colors = self.cmap(np.mod(self.labels[self._disp], self.cmap.N))
        self._scatter = self.ax.scatter(
            self.points[self._disp, 0], self.points[self._disp, 1],
            c=colors, s=4, alpha=0.35, linewidths=0, zorder=2,
        )

        if self.partitioner is not None:
            for index, (mean, width, height, angle) in enumerate(
                self.partitioner.ellipses()
            ):
                color = self.cmap(index % self.cmap.N)
                ellipse = EllipsePatch(
                    mean, width, height, angle=angle,
                    fill=False, edgecolor=color, lw=2, zorder=4,
                )
                self.ax.add_patch(ellipse)
                self._ellipses.append(ellipse)

        self._update_counts()
        self.fig.canvas.draw_idle()

    def _update_counts(self) -> None:
        classes, counts = np.unique(self.labels, return_counts=True)
        lines = ['class   count']
        lines += [f'{cls:>5}   {count:>7}' for cls, count in zip(classes, counts)]
        self.counts_text.set_text('\n'.join(lines))

    # --------------------------------------------------------------------- run
    def run(self) -> Optional[np.ndarray]:
        """Show the window and block until closed.

        Returns the ``(N,)`` label array if the user saved, else ``None``.
        """
        plt.show()
        if not self.saved:
            return None
        return self.labels
