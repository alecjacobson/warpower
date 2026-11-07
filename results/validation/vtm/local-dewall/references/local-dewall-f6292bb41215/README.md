# Local DeWall

CUDA/C++ implementation of **Local DeWall**, a memory-efficient GPU algorithm for constructing exact 3D Delaunay triangulations. The method represents a local Delaunay wall as an ordered point list and traverses adjacent tetrahedra incrementally, avoiding the explicit storage and repeated update of complete local one-rings.

## Paper

Wuheng Gao and Renjie Chen, "Local DeWall: Billion-scale 3D Delaunay triangulation on a single GPU," *Computer-Aided Design*, vol. 199, article 104107, 2026. [Paper page](https://doi.org/10.1016/j.cad.2026.104107)

If you use this code, please cite the paper above.

## Requirements and build

- Windows x64
- Visual Studio 2022 (MSVC v143)
- NVIDIA CUDA Toolkit 12.6
- An NVIDIA CUDA-capable GPU

Open `Delaunay3dGPU.sln`, select `Release | x64`, and build. The supplied project targets CUDA architecture `sm_86`; change **CUDA C/C++ > Device > Code Generation** when building for another GPU architecture.

## Usage

```text
Delaunay3dGPU.exe <output-prefix> <input.txt>
```

Example:

```text
Delaunay3dGPU.exe result_ points.txt
```

The input is a text file whose first line is the number of points, followed by one 3D point per line:

```text
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
```

Input coordinates are normalized automatically. The example produces:

- `result_x.bin`: sorted point coordinates (`float`, three values per point);
- `result_t.bin`: zero-based tetrahedron vertex indices (`int32`, four values per tetrahedron), referring to `result_x.bin`.

Each binary file starts with two `uint64` values storing its row and column counts, followed by the matrix data.

Running without an input file uses the built-in fixed-seed generator for 10 million uniformly distributed points. This repository is a research prototype corresponding to the Local DeWall method; parameter limits and CUDA architecture settings may need adjustment for other hardware or point distributions.

## License

This project is released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution of the robust geometric predicate routines.
