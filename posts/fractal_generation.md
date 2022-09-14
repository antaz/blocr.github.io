---
title: "Fractal Generation: A Rustacean's Prespective"
date: October, 1st 2021
---

The Mandelbrot Case Study
-------------------------

---

The mandelbrot is arguably the most popular fractal out there. The corresponding set is defined as follows

$$z_{n+1} = z_{n}^2 + c$$

In order to generate a fractal from this equation we plot all of the points $c$ in the complex plane that have a stable orbit under the iteration of $z_n$.

The following is an implementation of the Level-Set Method (LSM/M) or also known as the [escape time algorithm](https://en.wikipedia.org/wiki/Mandelbrot_set#Computer_drawings) in Rust

```Rust
// maximum iteration count
const MAX_ITER: i32 = 1000;
// escape radius squared
const RADIUS_SQ: f64 = 4.0;

// scalar implementation of Level Set Method
pub fn lsm(cr: f64, ci: f64) -> i32 {
    // z (real and imaginary parts init)
    let mut zr = 0.0;
    let mut zi = 0.0;
    // z^2 (real and imaginary parts init)
    let mut zr2 = 0.0;
    let mut zi2 = 0.0;
    // iteration count
    let mut iteration = 0;

    while (iteration < MAX_ITER) && (zr2 + zi2 < RADIUS_SQ) {
        // update z
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        // update z^2
        zr2 = zr * zr;
        zi2 = zi * zi;
        // and update the iteration count
        iteration = iteration + 1;
    }
    iteration
}
```

This implementation uses an interesting property of the set, namely a point $c$ belongs to the set if and only if $\vert z_n \vert  \leq{2}$ for all $n \geq{0}$. We call this bound on the $z$ values the escape radius.

To draw the mandelbrot set we use an image extension called **PPM** or the [portable pixmap format](http://netpbm.sourceforge.net/doc/ppm.html). The good thing about `PPM` is that it's a very simple ASCII based format with no compression.

The format looks something like this but not necessarily

```
P6           # a 'magic number to identify the ppm format'
800 600      # width and height of the pixel map
255          # maximum value of each color pixel
# The part above is the header
# The part below is the image data: RGB triplets
255   0   0  # red
  0 255   0  # green
  0   0 255  # blue
255 255   0  # yellow
255 255 255  # white
  0   0   0  # black
```

Finally, we plot the pixels and save the file with this format.

```Rust
fn render_mandelbrot() -> Vec<u8> {
    let mut img_buffer: Vec<u8> = vec![0; WIDTH * HEIGHT * 3];

    // main loop through all the pixels
    for y in 0..HEIGHT as u32 {
        for x in 0..WIDTH as u32 {
            // mapping the pixel coordinates to the Mandelbrot domain
            let (cr, ci) = (
                (x as f64 / WIDTH as f64) * (XMAX - XMIN) + XMIN,
                (y as f64 / HEIGHT as f64) * (YMAX - YMIN) + YMIN,
            );
            // calculate iterations
            let iterations = lsm(cr, ci);

            // set the pixels according to the iterations count
            let pixel_r = (y as usize * WIDTH + x as usize) * 3;
            let pixel_g = (y as usize * WIDTH + x as usize) * 3 + 1;
            let pixel_b = (y as usize * WIDTH + x as usize) * 3 + 2;

            if iterations == MAX_ITER {
                img_buffer[pixel_r] = 0;
                img_buffer[pixel_g] = 0;
                img_buffer[pixel_b] = 0;
            } else {
                img_buffer[pixel_r] = 255;
                img_buffer[pixel_g] = 255;
                img_buffer[pixel_b] = 255;
            }
        }
    }
    img_buffer
}
```

The function iterates through the screen coordinates, maps them to the range and domain of the mandelbrot function and then colors with black points that belong to the set and with white otherwise.

![1280x720p Mandelbrot fractal (converted to PNG)](./output/fractal.png)

Coloring
--------
I mean okay cool, but this is kinda boring and there's nothing to kill this boredom better than some coloring.
Mandelbrot coloring and fractal coloring in general is an exciting topic but it can get quite complicated very quickly.

We're going to use the iteration count that is returned from the `lsm()` function as an index into an arbitrarily chosen color palette.

![color palette](./output/color.png)

This color palette contains only 5 colors, but our iteration count function goes beyond that to 1000, so we truncate this using the module operator.

```Rust
if iterations == MAX_ITER {
    img_buffer[pixel_r] = 0;
    img_buffer[pixel_g] = 0;
    img_buffer[pixel_b] = 0;
} else {
    img_buffer[pixel_r] = palette[iterations as usize % palette.len()].0;
    img_buffer[pixel_g] = palette[iterations as usize % palette.len()].1;
    img_buffer[pixel_b] = palette[iterations as usize % palette.len()].2;
}
```

We get something slightly less boring than the black and white version but it could be better.

![Colored mandelbrot set](./output/fractal2.png)

Much better results can be achieved by using a larger palette of colors or by interpolating the colors into a spectrum of size `MAX_ITER`.


Optimization
------------
Rendering a single mandelbrot frame at a time is not a very CPU intensive process. To put things into perspective it takes around **~1.2** seconds to render the colored image above on my hardware setup, However when we want to generate a lot of frames with a limited time constraint this average is relatively large.

The bottleneck of this program's performance is the `lsm()` iteration function. We can confirm that from this performance flamegraph where the function takes **~86%** of the entire computation time, So that should be an interesting spot for optimization.

![flamegraph of the rendering program generated from the `perf-record(1)` output](./output/flamegraph.svg){width=100%}

### Parallelization
The process we've used so far is multithreading-friendly. However there are still some [race condition](https://en.wikipedia.org/wiki/Race_condition#Data_race) gotchas to take care of when implementing this in virtually any programming language. To avoid these race shenanigans we use a Rust crate for parallelization called [**Rayon**](https://lib.rs/crates/rayon).

Rayon uses a policy known as [Work Stealing](https://en.wikipedia.org/wiki/Work_stealing) that was first introduced in the [Cilk](https://en.wikipedia.org/wiki/Cilk) project.

In order to use Rayon, we need to turn the main loop in the `render_mandelbrot()` function into a **[parallel iterator](https://docs.rs/rayon/1.5.1/rayon/iter/index.html)**

```Rust
fn render_parallel_mandelbrot(palette: Vec<(u8, u8, u8)>) -> Vec<u8> {
    let mut img_buffer: Vec<u8> = vec![0; WIDTH * HEIGHT * 3];

    img_buffer
        .par_chunks_exact_mut(WIDTH * 3)
        .enumerate()
        .for_each(|(y, rows)| {
            rows.chunks_exact_mut(3)
                .enumerate()
                .for_each(|(x, triplet)| {
                    let (cr, ci) = (
                        (x as f64 / WIDTH as f64) * (XMAX - XMIN) + XMIN,
                        (y as f64 / HEIGHT as f64) * (YMAX - YMIN) + YMIN,
                    );
                    let iterations = lsm(cr, ci);

                    if iterations == MAX_ITER {
                        triplet[0] = 0;
                        triplet[1] = 0;
                        triplet[2] = 0;
                    } else {
                        triplet[0] = palette[iterations as usize % palette.len()].0;
                        triplet[1] = palette[iterations as usize % palette.len()].1;
                        triplet[2] = palette[iterations as usize % palette.len()].2;
                    }
                });
        });

    img_buffer
}
```
We divided our image buffer into `WIDTH` parallel chunks of pixel triplets and applied similar routine to the non-parallel version

#### Benchmark Results
Using the linux **perf-stat(1)** utility we can run 10 samples of the benchmark tests with the following command:

`perf stat -r 10 -ddd target/release/binary`

Version          CPU utilized  Instructions (B)  Time elapsed (s) 
---------------  ------------  ----------------  ---------------- 
non-parallel     0.999         3.505             1.204244 
parallel         3.839         3.485             0.36724 

Table: Results from my Core i3

The parallel version is **~4x** faster which is the expected theoretical improvement of using the rest of the 3 idling cores.

### Vectorization (SIMD)
Or also known as playing with all cards. [SIMD](https://en.wikipedia.org/wiki/SIMD) stands for Single Instruction Multiple Data which is a way to exploit the data-level parallelism on the CPU. It provides an extended set of registers and instructions that can operate on varying sets of data vectors at a time, such as 128, 256 or even 512-bits.

One of the most commonly supported extensions is [**AVX2**](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) (Advanced Vector Extension) on Intel and AMD, which allows us to work with 256-bit register operations, or in Rust jargon, four `f64`s (double-precision floating point numbers).

Modern compilers including the Rust compiler will often try *auto vectorizing* your code, meaning that it will try to optimize the code to use SIMD, However this is not always possible due to some set of limitations, in which case SIMD can be enforced by writing the *intrinsics instructions* by hand.

Rust supports a wide variety of SIMD extensions for several ISAs (Instruction Set Architectures). The documentation of this can be found in the [core::arch](https://doc.rust-lang.org/core/arch/index.html) module.

Let's try to write a vectorized version of our `lsm()` function 

```Rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[target_feature(enable = "avx2")]
pub unsafe fn lsm_avx2(cr: __m256d, ci: __m256d) -> __m256d {
    // z (real and imaginary parts init)
    let mut zr = _mm256_set1_pd(0.0);
    let mut zi = _mm256_set1_pd(0.0);

    // z^2 (real and imaginary parts init)
    let mut zr2 = _mm256_set1_pd(0.0);
    let mut zi2 = _mm256_set1_pd(0.0);

    // useful constants
    let one = _mm256_set1_pd(1.0);
    let two = _mm256_set1_pd(2.0);
    let four = _mm256_set1_pd(4.0);

    // iteration counts
    let mut iterations = _mm256_set1_pd(0.0);

    for _ in 0..MAX_ITER {
        // comparison mask of the magnitudes with the escape radius
        let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(_mm256_add_pd(zr2, zi2), four);

        // update the iteration counts
        iterations = _mm256_add_pd(_mm256_and_pd(mask, one), iterations);

        // break if all values exceeded the threshold
        if _mm256_movemask_pd(mask) == 0 {
            break;
        }

        // update z
        zi = _mm256_add_pd(_mm256_mul_pd(two, _mm256_mul_pd(zr, zi)), ci);
        zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);

        // update z^2
        zr2 = _mm256_mul_pd(zr, zr);
        zi2 = _mm256_mul_pd(zi, zi);
    }
    iterations
}
```

First we include the `std::arch` module to use the provided data types and the corresponding functions, then tell the compiler that we're targeting processors supporting the AVX2 extension.
The arguments are `__m256d`, which is a 256-bit wide set of four `f64` types. The rest of the function can be translated easily with the scalar version `lsm()` with the only exception that we're working with four `f64`s instead of one.

#### Under the hood
Before we benchmark the AVX2 implementation, and for what it's worth, we compare the disassembly output of both [`lsm()`](https://gist.github.com/blocr/0dbe515b117a32d462d3b8c704686688) and [`lsm_avx2()`](https://gist.github.com/blocr/fbf1e5158ccdd9972f917f4e9883c1f2)

Notice that the Rust compiler automatically optimizes the scalar version of the LSM function by using an extension knowns as [**SSE2**](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions)

One more thing to add to this. We divide the image buffer further to process 4 chunks of pixels at a time in the `render_parallel_mandelbrot()` function

```Rust
img_buffer
    .par_chunks_exact_mut(WIDTH * 3)
    .enumerate()
    .for_each(|(y, rows)| {
        rows.chunks_exact_mut(12)
            .enumerate()
            .for_each(|(c, chunk)| {
                let c = (c as f64) * 4.0;
                let y = y as f64;

                let cr = &mut [c, c + 1.0, c + 2.0, c + 3.0];
                let ci = &mut [y; 4];

                for (cr, ci) in cr.iter_mut().zip(ci.iter_mut()) {
                    *cr = (*cr / WIDTH as f64) * (XMAX - XMIN) + XMIN;
                    *ci = (*ci / HEIGHT as f64) * (YMAX - YMIN) + YMIN;
                }

                let iterations: [f64; 4] =
                    unsafe { transmute(lsm_avx2(transmute(*cr), transmute(*ci))) };
                chunk
                    .chunks_exact_mut(3)
                    .enumerate()
                    .for_each(|(t, triplet)| {
                        if iterations[t] == MAX_ITER as f64 {
                            triplet[0] = 0;
                            triplet[1] = 0;
                            triplet[2] = 0;
                        } else {
                            triplet[0] = palette[iterations[t] as usize % palette.len()].0;
                            triplet[1] = palette[iterations[t] as usize % palette.len()].1;
                            triplet[2] = palette[iterations[t] as usize % palette.len()].2;
                        }
                    })
            });
    });
```

#### Back to Benchmarks

Version          CPU utilized  Instructions (B)  Time elapsed (s) 
---------------  ------------  ----------------  ---------------- 
non-parallel     0.999         3.505             1.204244 
parallel         3.839         3.485             0.36724 
parallel-avx2    3.792         0.864             0.121628 


The last version is **3x** faster, and we have a significant decrease on the number of instructions performed, but what happened here? Theoretically, this should have been **4x** faster. The reality is that taking full advantage of SIMD capability is not always straightforward, some latency can emerge due to [data-alignment](https://en.wikipedia.org/wiki/Data_structure_alignment) problems or instruction throughput and latency, but for the purposes of this discussion, i think it's fair to be satisfied with this suboptimal result.

#### Suggested Readings

---
- [The Science of Fractal Images](https://www.springer.com/gp/book/9781461283492)
- [Achieving warp speed with Rust](https://gist.github.com/jFransham/369a86eff00e5f280ed25121454acec1)
