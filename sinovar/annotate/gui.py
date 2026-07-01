"""Interactive matplotlib GUI for annotating the 2D embedding plane.

The GUI shows the 2D histogram of the (reduced) embedding and lets the user
partition it into classes with a Gaussian-mixture model. Three ways to drive
the partition are offered:

* **Fit K classes** — automatic GMM for the number of classes on the slider.
* **Auto-K (BIC)** — automatic GMM whose number of classes minimises the BIC.
* **Seed mode / Fit seeds** — manual GMM initialised from clicked seed points.

The partition is exclusive and exhaustive, so every particle always belongs to
exactly one class. On start-up all particles share a single class, preserving
that invariant even before the user acts.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse as EllipsePatch
from matplotlib.widgets import Button, Slider

from .partition import GmmPartitioner, fit_gmm_bic

logger = logging.getLogger(__name__)

#: Cap on the number of points drawn in the scatter overlay, for responsiveness.
_MAX_SCATTER = 50_000


class AnnotationApp:
    """Interactive annotation of a 2D point cloud into exclusive classes."""

    def __init__(
        self,
        points: np.ndarray,
        bins: int = 200,
        initial_classes: int = 3,
        max_classes: int = 20,
        cmap: str = 'tab20',
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must be an (N, 2) array')

        self.points = points
        self.n = points.shape[0]
        self.bins = bins
        self.initial_classes = int(np.clip(initial_classes, 1, max_classes))
        self.max_classes = max_classes
        self.cmap = plt.get_cmap(cmap)

        # Every particle starts in a single class -> partition invariant holds.
        self.labels = np.zeros(self.n, dtype=np.int64)
        self.partitioner: Optional[GmmPartitioner] = None
        self.seeds: List[Tuple[float, float]] = []
        self.seed_mode = False
        self.saved = False

        # Subsample the scatter overlay for large datasets (labels stay full).
        if self.n > _MAX_SCATTER:
            rng = np.random.default_rng(0)
            self._disp = rng.choice(self.n, _MAX_SCATTER, replace=False)
        else:
            self._disp = np.arange(self.n)

        self._scatter = None
        self._seed_artist = None
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
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._redraw()

    def _build_controls(self) -> None:
        # Widget axes are placed in figure coordinates in the right-hand column.
        left, width = 0.80, 0.17

        self.ax_slider = self.fig.add_axes([left, 0.86, width, 0.03])
        self.slider_k = Slider(
            self.ax_slider, 'K', 1, self.max_classes,
            valinit=self.initial_classes, valstep=1,
        )

        def button(y: float, label: str, callback) -> Button:
            axis = self.fig.add_axes([left, y, width, 0.055])
            widget = Button(axis, label)
            widget.on_clicked(callback)
            return widget

        self.btn_fit_k = button(0.75, 'Fit K classes', self._on_fit_k)
        self.btn_auto_k = button(0.68, 'Auto-K (BIC)', self._on_auto_k)
        self.btn_seed = button(0.58, 'Seed mode: OFF', self._on_toggle_seed)
        self.btn_fit_seeds = button(0.51, 'Fit seeds', self._on_fit_seeds)
        self.btn_clear_seeds = button(0.44, 'Clear seeds', self._on_clear_seeds)
        self.btn_clear_annotation = button(
            0.37, 'Clear annotation', self._on_clear_annotation
        )
        self.btn_save = button(0.30, 'Save & close', self._on_save)

        self.status_text = self.fig.text(
            left, 0.22, '', fontsize=9, va='top', wrap=True,
        )
        self.counts_text = self.fig.text(
            left, 0.16, '', fontsize=8, va='top', family='monospace',
        )

    # -------------------------------------------------------------- callbacks
    def _on_fit_k(self, _event) -> None:
        k = int(self.slider_k.val)
        self._apply(GmmPartitioner(n_components=k), f'Fitted GMM, K={k}')

    def _on_auto_k(self, _event) -> None:
        self._status('Searching K by BIC...')
        self.fig.canvas.draw_idle()
        best = fit_gmm_bic(self.points, range(1, self.max_classes + 1))
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
            'Click the histogram to place seeds'
            if self.seed_mode else 'Seed mode off'
        )
        self.fig.canvas.draw_idle()

    def _on_fit_seeds(self, _event) -> None:
        if not self.seeds:
            self._status('Place at least one seed first')
            self.fig.canvas.draw_idle()
            return
        means = np.asarray(self.seeds, dtype=np.float64)
        self._apply(
            GmmPartitioner(n_components=len(self.seeds), means_init=means),
            f'Fitted GMM from {len(self.seeds)} seed(s)',
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

    def _on_click(self, event) -> None:
        if not self.seed_mode or event.inaxes is not self.ax:
            return
        # Ignore clicks while a navigation tool (zoom/pan) is active.
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return
        self.seeds.append((event.xdata, event.ydata))
        self._status(f'{len(self.seeds)} seed(s) placed')
        self._draw_seeds()

    # ----------------------------------------------------------------- helpers
    def _apply(self, partitioner: GmmPartitioner, message: str) -> None:
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
        if self.seeds:
            seeds = np.asarray(self.seeds)
            self._seed_artist = self.ax.scatter(
                seeds[:, 0], seeds[:, 1],
                marker='x', c='red', s=90, linewidths=2, zorder=5,
            )
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
