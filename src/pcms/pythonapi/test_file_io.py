#!/usr/bin/env python3
"""
Test Omega_h file I/O Python bindings
"""

import pcms
import PyOmega_h as omega_h
import numpy as np
import os
import shutil
import gc
import tempfile

def test_binary_io(lib, world):
    """Test binary file format I/O"""
    print("\n=== Testing Binary Format I/O ===")
    
    # Build a simple 3D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,  # x, y, z dimensions
        4, 4, 4,         # nx, ny, nz divisions
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_binary_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test binary write/read
        binary_file = os.path.join(test_dir, "test_mesh.osh")
        print(f"Writing to binary file: {binary_file}")
        omega_h.write_mesh_binary(binary_file, mesh)
        
        print(f"Reading from binary file: {binary_file}")
        mesh_read = omega_h.read_mesh_binary(binary_file, lib)
        
        # Verify mesh properties
        assert mesh_read.dim() == dim_orig, f"Dimension mismatch: {mesh_read.dim()} != {dim_orig}"
        assert mesh_read.nverts() == nverts_orig, f"Vertex count mismatch: {mesh_read.nverts()} != {nverts_orig}"
        assert mesh_read.nelems() == nelems_orig, f"Element count mismatch: {mesh_read.nelems()} != {nelems_orig}"
        
        print("✓ Binary I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh_read
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_gmsh_io(lib, world):
    """Test Gmsh file format I/O"""
    print("\n=== Testing Gmsh Format I/O ===")
    
    # Build a simple 2D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        2.0, 1.0, 0.0,  # x, y, z dimensions (z=0 for 2D)
        5, 3, 0,         # nx, ny, nz divisions (nz=0 for 2D)
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_gmsh_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test Gmsh write/read
        gmsh_file = os.path.join(test_dir, "test_mesh.msh")
        print(f"Writing to Gmsh file: {gmsh_file}")
        omega_h.write_mesh_gmsh(gmsh_file, mesh)
        
        print(f"Reading from Gmsh file: {gmsh_file}")
        mesh_read = omega_h.read_mesh_gmsh(gmsh_file, world)
        
        # Verify mesh properties
        assert mesh_read.dim() == dim_orig, f"Dimension mismatch: {mesh_read.dim()} != {dim_orig}"
        assert mesh_read.nverts() == nverts_orig, f"Vertex count mismatch: {mesh_read.nverts()} != {nverts_orig}"
        assert mesh_read.nelems() == nelems_orig, f"Element count mismatch: {mesh_read.nelems()} != {nelems_orig}"
        
        print("✓ Gmsh I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh_read
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_vtk_io(lib, world):
    """Test VTK file format I/O"""
    print("\n=== Testing VTK Format I/O ===")
    
    # Build a simple 3D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.5, 1.0, 0.5,  # x, y, z dimensions
        3, 3, 2,         # nx, ny, nz divisions
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    coords_orig = mesh.coords()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    print(f"Coordinates shape: {coords_orig.shape}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_vtk_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test VTU write (VTU is write-only in the API, typically for visualization)
        vtu_file = os.path.join(test_dir, "test_mesh.vtu")
        print(f"Writing to VTU file: {vtu_file}")
        omega_h.write_mesh_vtu(vtu_file, mesh, compress=False)
        
        # Verify file was created
        assert os.path.exists(vtu_file), f"VTU file was not created: {vtu_file}"
        file_size = os.path.getsize(vtu_file)
        print(f"VTU file created successfully (size: {file_size} bytes)")
        
        # Test compressed VTU write
        vtu_compressed = os.path.join(test_dir, "test_mesh_compressed.vtu")
        print(f"Writing compressed VTU file: {vtu_compressed}")
        omega_h.write_mesh_vtu(vtu_compressed, mesh, compress=True)
        
        assert os.path.exists(vtu_compressed), f"Compressed VTU file was not created"
        compressed_size = os.path.getsize(vtu_compressed)
        print(f"Compressed VTU file created (size: {compressed_size} bytes)")
        
        print("✓ VTK I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_meshb_io(lib, world):
    """Test MESHB file format I/O"""
    print("\n=== Testing MESHB Format I/O ===")
    
    # Check if MESHB support is available
    if not hasattr(omega_h, 'write_mesh_meshb'):
        print("⊘ MESHB support not available (OMEGA_H_USE_LIBMESHB not enabled)")
        return
    
    # Build a simple 3D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,  # x, y, z dimensions
        3, 3, 3,         # nx, ny, nz divisions
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_meshb_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test MESHB write/read
        meshb_file = os.path.join(test_dir, "test_mesh.mesh")
        print(f"Writing to MESHB file: {meshb_file}")
        omega_h.write_mesh_meshb(mesh, meshb_file, version=2)
        
        print(f"Reading from MESHB file: {meshb_file}")
        # MESHB read requires pre-created mesh object
        mesh_read = omega_h.Mesh(lib)
        mesh_read.set_comm(world)
        omega_h.read_mesh_meshb(mesh_read, meshb_file)
        
        # Verify mesh properties
        assert mesh_read.dim() == dim_orig, f"Dimension mismatch: {mesh_read.dim()} != {dim_orig}"
        assert mesh_read.nverts() == nverts_orig, f"Vertex count mismatch: {mesh_read.nverts()} != {nverts_orig}"
        assert mesh_read.nelems() == nelems_orig, f"Element count mismatch: {mesh_read.nelems()} != {nelems_orig}"
        
        print("✓ MESHB I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh_read
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_exodus_io(lib, world):
    """Test Exodus file format I/O"""
    print("\n=== Testing Exodus Format I/O ===")
    
    # Check if Exodus support is available
    if not hasattr(omega_h, 'write_mesh_exodus'):
        print("⊘ Exodus support not available (OMEGA_H_USE_SEACASEXODUS not enabled)")
        return
    
    # Build a simple 3D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,  # x, y, z dimensions
        3, 3, 3,         # nx, ny, nz divisions
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_exodus_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test Exodus write
        exodus_file = os.path.join(test_dir, "test_mesh.exo")
        print(f"Writing to Exodus file: {exodus_file}")
        omega_h.write_mesh_exodus(exodus_file, mesh, verbose=False)
        
        # Test Exodus file handle API
        if hasattr(omega_h, 'exodus_open'):
            print(f"Testing Exodus file handle API with: {exodus_file}")
            
            # Open the file
            exo_handle = omega_h.exodus_open(exodus_file, verbose=False)
            print(f"Opened Exodus file with handle: {exo_handle}")
            
            # Get number of time steps
            num_steps = omega_h.exodus_get_num_time_steps(exo_handle)
            print(f"Number of time steps: {num_steps}")
            
            # Read mesh using file handle
            mesh_from_handle = omega_h.OmegaHMesh(lib)
            mesh_from_handle.set_comm(world)
            omega_h.read_mesh_exodus(exo_handle, mesh_from_handle, verbose=False)
            
            # Verify mesh properties
            assert mesh_from_handle.dim() == dim_orig, f"Dimension mismatch: {mesh_from_handle.dim()} != {dim_orig}"
            assert mesh_from_handle.nverts() == nverts_orig, f"Vertex count mismatch: {mesh_from_handle.nverts()} != {nverts_orig}"
            assert mesh_from_handle.nelems() == nelems_orig, f"Element count mismatch: {mesh_from_handle.nelems()} != {nelems_orig}"
            
            # Close the file
            omega_h.exodus_close(exo_handle)
            # Explicitly delete mesh object from handle
            del mesh_from_handle
        
        print("✓ Exodus I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_adios2_io(lib, world):
    """Test ADIOS2 file format I/O"""
    print("\n=== Testing ADIOS2 Format I/O ===")
    
    # Check if ADIOS2 support is available
    if not hasattr(omega_h, 'write_mesh_adios2'):
        print("⊘ ADIOS2 support not available (OMEGA_H_USE_ADIOS2 not enabled)")
        return
    
    # Build a simple 3D box mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,  # x, y, z dimensions
        3, 3, 3,         # nx, ny, nz divisions
        False            # symmetric
    )
    
    # Get initial mesh properties
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    dim_orig = mesh.dim()
    print(f"Original mesh: dim={dim_orig}, nverts={nverts_orig}, nelems={nelems_orig}")
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_adios2_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Test ADIOS2 write/read
        adios2_file = os.path.join(test_dir, "test_mesh.bp")
        print(f"Writing to ADIOS2 file: {adios2_file}")
        omega_h.write_mesh_adios2(adios2_file, mesh, prefix="")
        
        print(f"Reading from ADIOS2 file: {adios2_file}")
        mesh_read = omega_h.read_mesh_adios2(adios2_file, lib, prefix="")
        
        # Verify mesh properties
        assert mesh_read.dim() == dim_orig, f"Dimension mismatch: {mesh_read.dim()} != {dim_orig}"
        assert mesh_read.nverts() == nverts_orig, f"Vertex count mismatch: {mesh_read.nverts()} != {nverts_orig}"
        assert mesh_read.nelems() == nelems_orig, f"Element count mismatch: {mesh_read.nelems()} != {nelems_orig}"
        
        print("✓ ADIOS2 I/O test passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh_read
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")


def test_read_mesh_file_auto_detect(lib, world):
    """Test automatic format detection with read_mesh_file"""
    print("\n=== Testing Automatic Format Detection ===")
    
    # Build a test mesh
    print("Creating test mesh...")
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        3, 3, 3,
        False
    )
    
    nverts_orig = mesh.nverts()
    nelems_orig = mesh.nelems()
    
    # Create unique temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="pcms_test_autodetect_")
    print(f"Using temporary directory: {test_dir}")
    try:
        # Write in different formats
        binary_file = os.path.join(test_dir, "mesh.osh")
        gmsh_file = os.path.join(test_dir, "mesh.msh")
        
        omega_h.write_mesh_binary(binary_file, mesh)
        omega_h.write_mesh_gmsh(gmsh_file, mesh)
        
        # Test auto-detection for binary format
        print(f"Auto-detecting binary file: {binary_file}")
        mesh_binary = omega_h.read_mesh_file(binary_file, world)
        assert mesh_binary.nverts() == nverts_orig
        assert mesh_binary.nelems() == nelems_orig
        print("✓ Binary auto-detection passed")
        
        # Test auto-detection for Gmsh format
        print(f"Auto-detecting Gmsh file: {gmsh_file}")
        mesh_gmsh = omega_h.read_mesh_file(gmsh_file, world)
        assert mesh_gmsh.nverts() == nverts_orig
        assert mesh_gmsh.nelems() == nelems_orig
        print("✓ Gmsh auto-detection passed")
        
        # Explicitly delete mesh objects before cleanup
        del mesh_binary
        del mesh_gmsh
        del mesh
        gc.collect()  # Force garbage collection
        
    finally:
        # Clean up temporary files
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("Omega_h File I/O Python Binding Tests")
    print("=" * 60)
    
    # Create library and world communicator once for all tests
    lib = omega_h.OmegaHLibrary()
    world = lib.world()
    
    try:
        test_binary_io(lib, world)
        test_gmsh_io(lib, world)
        test_vtk_io(lib, world)
        test_meshb_io(lib, world)
        test_exodus_io(lib, world)
        test_adios2_io(lib, world)
        test_read_mesh_file_auto_detect(lib, world)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        # Explicitly delete objects to avoid an MPI finalizing issue
        del world
        del lib