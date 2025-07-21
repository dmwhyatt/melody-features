#!/usr/bin/env python3
"""
Test script to verify IDyOM standalone setup
"""

import sys
import os
import subprocess
import importlib.util

def test_python_dependencies():
    """Test that all required Python packages are available"""
    print("Testing Python dependencies...")
    
    required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'natsort']
    missing_packages = []
    
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
            else:
                print(f"  ✓ {package} found")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"  ✗ Missing packages: {missing_packages}")
        print("  Run: pip install -r requirements.txt")
        return False
    else:
        print("  ✓ All Python dependencies found")
        return True

def test_system_dependencies():
    """Test that SBCL and SQLite are available"""
    print("Testing system dependencies...")
    
    # Test SBCL
    try:
        result = subprocess.run(['sbcl', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ SBCL found: {result.stdout.strip()}")
        else:
            print("  ✗ SBCL not found or not working")
            return False
    except FileNotFoundError:
        print("  ✗ SBCL not found. Install with: brew install sbcl (macOS) or apt-get install sbcl (Ubuntu)")
        return False
    
    # Test SQLite
    try:
        result = subprocess.run(['sqlite3', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ SQLite found: {result.stdout.strip()}")
        else:
            print("  ✗ SQLite not found or not working")
            return False
    except FileNotFoundError:
        print("  ✗ SQLite not found. Install with: brew install sqlite3 (macOS) or apt-get install sqlite3 (Ubuntu)")
        return False
    
    return True

def test_idyom_installation():
    """Test that IDyOM is properly installed"""
    print("Testing IDyOM installation...")
    
    # Check if IDyOM directories exist
    idyom_path = os.path.expanduser("~/idyom")
    if not os.path.exists(idyom_path):
        print(f"  ✗ IDyOM directory not found at {idyom_path}")
        print("  Run: ./setup_idyom.sh")
        return False
    
    # Check if database exists
    db_path = os.path.join(idyom_path, "db", "database.sqlite")
    if not os.path.exists(db_path):
        print(f"  ✗ IDyOM database not found at {db_path}")
        print("  Run: ./setup_idyom.sh")
        return False
    
    # Try to connect to IDyOM
    try:
        lisp_test = '(start-idyom)'
        result = subprocess.run([
            'sbcl', '--non-interactive', 
            '--eval', lisp_test,
            '--eval', '(quit)'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✓ IDyOM can be started successfully")
            return True
        else:
            print("  ✗ IDyOM failed to start")
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ✗ IDyOM startup timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error testing IDyOM: {e}")
        return False

def test_py2lisp_module():
    """Test that py2lisp module can be imported"""
    print("Testing py2lisp module...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Try importing the main components
        from py2lisp.run import IDyOMExperiment
        from py2lisp.configuration import IDyOMConfiguration
        from py2lisp.export import Export
        
        print("  ✓ py2lisp modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import py2lisp modules: {e}")
        return False

def test_data_availability():
    """Test that sample data is available"""
    print("Testing data availability...")
    
    test_data_path = "data/small_kern_subset/"
    if not os.path.exists(test_data_path):
        print(f"  ✗ Test data not found at {test_data_path}")
        return False
    
    # Check for .krn files
    krn_files = [f for f in os.listdir(test_data_path) if f.endswith('.krn')]
    if not krn_files:
        print(f"  ✗ No .krn files found in {test_data_path}")
        return False
    
    print(f"  ✓ Found {len(krn_files)} .krn files in test dataset")
    return True

def main():
    """Run all tests"""
    print("IDyOM Standalone Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("System Dependencies", test_system_dependencies),
        ("IDyOM Installation", test_idyom_installation),
        ("py2lisp Module", test_py2lisp_module),
        ("Test Data", test_data_availability),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run example analysis: python example_analysis.py")
        print("2. Or use the command line: python idyom_analyzer.py data/small_kern_subset/ --texture melody")
        return True
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 