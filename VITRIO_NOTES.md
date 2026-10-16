# Notes on vitrio from porting sinovar to it

Written while replacing sinovar's own reader with vitrio 0.1.4. Everything
here is a suggestion from one caller, not a defect report: the port removed
about 300 lines from sinovar and the result is better than what it replaced.

## What worked without any friction

- `read_batch_async` into a caller-owned array is exactly the right shape for
  a tiled sweep. The caller keeps its own buffers, so double buffering is the
  caller's business and vitrio does not have to guess at a pool.
- `ImageLocation(key, index)` and `from_string` cover both the star-file
  spelling and the case where a path itself contains `@`, which the sinovar
  parser got wrong.
- Rejecting an indexed read into a volume caught a bug in my own test fixture,
  which was writing a 3D volume where it meant a stack.
- The three scratch constructors map one to one onto "none, memory, fast
  storage", which is the choice a user actually wants to express.

## Suggestions

### 1. A way to ask what a `Completion` is waiting on

When a batch fails, `get` raises the first failure. For a batch of 500
particles read from one stack that is enough, but when a batch spans files it
is useful to know which location failed, to name it in a message or to skip it
and carry on. Something like a `failures()` returning the indices and their
errors, with `get` unchanged, would cover it.

The concrete case: EMPIAR-10180 has one corrupt particle out of 327k. Today
finding it meant scanning the stack separately; with per-location failures the
read itself could report it.

### 2. `query_descriptor` for several files at once

Opening one file per call is fine for one stack. A data set spread over
thousands of per-micrograph stacks pays a round trip each. A batched form, or
one that takes the same `ImageLocationGrouping` the scratches take, would let
a caller validate a whole star file in one go.

### 3. Say in the docs whether a destination may be reused immediately

`read_batch_async` says the completion keeps `destination` alive and that
destroying it waits. What a caller needs to know for double buffering is the
other direction: once `get` has returned, is the buffer free to overwrite
while a later batch is in flight? I assumed yes and the tests agree, but it is
the one invariant the whole buffer rotation rests on, so it would be worth
stating.

### 4. A scratch that reports what it holds

`create_host_image_scratch` takes `max_size` and silently holds what fits.
A caller that asks for 200 GB on a 128 GB machine gets a partly warm scratch
and no way to say so in a log. A property for how many images or bytes it
actually took would let callers report "holding 60% of the data set" rather
than guessing.

### 5. Minor: `no_stack_index` has the wrong docstring

`ImageLocation.no_stack_index` shows `int([x]) -> integer`, which is `int`'s
docstring rather than its own.

## One thing I would not change

`loader()` owning its executor by default, with the option to pass one in, is
the right default. The note that two loaders mean two thread pools is easy to
miss, though, and a caller who builds a loader per command without noticing
pays for it.

## A bug this port found, and what would have prevented it

The first version of the driver drew destinations from a small pool of reused
host buffers, handing one back as soon as the tile kernel had been dispatched.
That is wrong: `jax.device_put` is asynchronous, and on the CPU backend it may
alias the array rather than copy it, so a later `read_batch_async` into the
same buffer overwrote images a tile was still reading. The result was a
distance matrix that differed by tens of percent between identical runs, while
every unit test passed, because the test images were small enough that each
read finished before the buffer came round again.

Nothing here is vitrio's fault; the contract is clear that the destination is
caller-owned. But two things would have made the mistake harder:

- Stating in `read_batch_async` whether the destination may be overwritten
  once `get` has returned (suggestion 3 above). A caller reasoning about a
  buffer pool needs that sentence and I could not find it.
- A debug mode that poisons or write-protects a destination between reads, so
  that a second read into a buffer still referenced elsewhere is caught at
  once rather than showing up as a wrong number much later.

The fix was to stop pooling and let every span own its array, which costs an
allocation per block and is far simpler to reason about.
