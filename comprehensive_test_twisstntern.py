#!/usr/bin/env python3
"""
Comprehensive Testing Suite for TWISSTNTERN
============================================

This script performs thorough testing of the twisstntern package including:
- Basic functionality with various input files
- Parameter testing (granularity, colormaps, downsampling)
- Edge case handling
- Performance testing
- Output validation
- Error handling

Author: AI Assistant
Date: 2025-07-03
"""

import os
import sys
import time
import traceback
import shutil
import glob
from pathlib import Path
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, '.')

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_test(test_name):
    """Print a formatted test header"""
    print(f"\n🧪 TEST: {test_name}")
    print("-" * 50)

def check_outputs(output_dir, expected_files):
    """Check if expected output files exist and are valid"""
    missing_files = []
    file_info = {}
    
    for expected_file in expected_files:
        file_path = Path(output_dir) / expected_file
        if file_path.exists():
            file_info[expected_file] = {
                'exists': True,
                'size': file_path.stat().st_size,
                'readable': True
            }
            # Try to read if it's a CSV
            if expected_file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    file_info[expected_file]['rows'] = len(df)
                    file_info[expected_file]['cols'] = len(df.columns)
                except Exception as e:
                    file_info[expected_file]['readable'] = False
                    file_info[expected_file]['error'] = str(e)
        else:
            missing_files.append(expected_file)
            file_info[expected_file] = {'exists': False}
    
    return missing_files, file_info

def run_basic_functionality_tests():
    """Test basic functionality with different input files"""
    print_section("BASIC FUNCTIONALITY TESTS")
    
    # Test files of different sizes
    test_files = [
        ("csv files/gene_flow_sims/A_m0_trimmed.csv", "Small file (~300 lines)"),
        ("csv files/gene_flow_sims/B_m0.001CORRECTED.csv", "Medium file (~500 lines)"),
        ("csv files/gene_flow_sims/A_m0.csv", "Large file (~30K lines)"),
        ("csv files/samples_14.6_testing_NN/A_m0_trimmed.csv", "Different dataset"),
    ]
    
    expected_outputs = [
        "fundamental_asymmetry.png",
        "granuality_0.1.png", 
        "heatmap.png",
        "radcount.png",
        "analysis_granularity_0.1.png",
        "index_granularity_0.1.png",
        "triangle_analysis_0.1.csv"
    ]
    
    results = {}
    
    for file_path, description in test_files:
        print_test(f"Basic analysis - {description}")
        
        if not os.path.exists(file_path):
            print(f"❌ SKIP: File not found: {file_path}")
            continue
            
        try:
            from twisstntern.pipeline import run_analysis
            
            start_time = time.time()
            
            # Run analysis
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=file_path,
                output_dir=f"test_output_basic_{Path(file_path).stem}"
            )
            
            duration = time.time() - start_time
            
            # Validate outputs
            output_dir = f"test_output_basic_{Path(file_path).stem}"
            prefix = Path(file_path).stem
            expected_files = [f"{prefix}_{f}" for f in expected_outputs]
            
            missing_files, file_info = check_outputs(output_dir, expected_files)
            
            # Validate results
            assert isinstance(triangles_results, pd.DataFrame), "triangles_results should be DataFrame"
            assert isinstance(fundamental_results, tuple), "fundamental_results should be tuple"
            assert len(fundamental_results) == 5, "fundamental_results should have 5 elements"
            assert isinstance(csv_file_used, str), "csv_file_used should be string"
            assert os.path.exists(csv_file_used), "CSV file should exist"
            
            results[file_path] = {
                'status': 'PASS',
                'duration': duration,
                'missing_files': missing_files,
                'triangles_shape': triangles_results.shape,
                'fundamental_results': fundamental_results,
                'csv_file': csv_file_used
            }
            
            print(f"✅ PASS: {description}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Triangles shape: {triangles_results.shape}")
            print(f"   Fundamental results: n_right={fundamental_results[0]}, n_left={fundamental_results[1]}, p-value={fundamental_results[4]:.2e}")
            print(f"   Missing files: {len(missing_files)}")
            
        except Exception as e:
            results[file_path] = {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(f"❌ FAIL: {description}")
            print(f"   Error: {str(e)}")
    
    return results

def run_parameter_tests():
    """Test different parameter combinations"""
    print_section("PARAMETER TESTING")
    
    test_file = "csv files/gene_flow_sims/B_m0.001CORRECTED.csv"
    if not os.path.exists(test_file):
        print(f"❌ SKIP: Test file not found: {test_file}")
        return {}
    
    results = {}
    
    # Test different granularities
    granularities = [0.05, 0.1, 0.2]
    
    for granularity in granularities:
        print_test(f"Granularity test - {granularity}")
        
        try:
            from twisstntern.pipeline import run_analysis
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=test_file,
                granularity=granularity,
                output_dir=f"test_output_granularity_{granularity}"
            )
            
            results[f"granularity_{granularity}"] = {
                'status': 'PASS',
                'triangles_shape': triangles_results.shape
            }
            print(f"✅ PASS: Granularity {granularity}")
            print(f"   Triangles shape: {triangles_results.shape}")
            
        except Exception as e:
            results[f"granularity_{granularity}"] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ FAIL: Granularity {granularity}")
            print(f"   Error: {str(e)}")
    
    # Test different colormaps
    colormaps = ["viridis", "viridis_r", "plasma", "inferno", "Blues", "Greys"]
    
    for colormap in colormaps:
        print_test(f"Colormap test - {colormap}")
        
        try:
            from twisstntern.pipeline import run_analysis
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=test_file,
                colormap=colormap,
                output_dir=f"test_output_colormap_{colormap}"
            )
            
            results[f"colormap_{colormap}"] = {
                'status': 'PASS'
            }
            print(f"✅ PASS: Colormap {colormap}")
            
        except Exception as e:
            results[f"colormap_{colormap}"] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ FAIL: Colormap {colormap}")
            print(f"   Error: {str(e)}")
    
    # Test downsampling
    downsample_tests = [
        (10, None, "Every 10th row"),
        (5, 2, "Every 5th row starting from index 2"),
        (20, 0, "Every 20th row starting from index 0")
    ]
    
    for downsample_N, downsample_i, description in downsample_tests:
        print_test(f"Downsampling test - {description}")
        
        try:
            from twisstntern.pipeline import run_analysis
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=test_file,
                downsample_N=downsample_N,
                downsample_i=downsample_i,
                output_dir=f"test_output_downsample_{downsample_N}_{downsample_i or 0}"
            )
            
            results[f"downsample_{downsample_N}_{downsample_i}"] = {
                'status': 'PASS',
                'triangles_shape': triangles_results.shape
            }
            print(f"✅ PASS: {description}")
            print(f"   Triangles shape: {triangles_results.shape}")
            
        except Exception as e:
            results[f"downsample_{downsample_N}_{downsample_i}"] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ FAIL: {description}")
            print(f"   Error: {str(e)}")
    
    return results

def run_edge_case_tests():
    """Test edge cases and error handling"""
    print_section("EDGE CASE TESTING")
    
    results = {}
    
    # Test with very small file
    print_test("Very small file test")
    small_file = "csv files/samples_14.6_testing_NN/I_m0.06_trimmed.csv"
    
    if os.path.exists(small_file):
        try:
            from twisstntern.pipeline import run_analysis
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=small_file,
                output_dir="test_output_very_small"
            )
            
            results['very_small_file'] = {
                'status': 'PASS',
                'triangles_shape': triangles_results.shape
            }
            print(f"✅ PASS: Very small file handled correctly")
            print(f"   Triangles shape: {triangles_results.shape}")
            
        except Exception as e:
            results['very_small_file'] = {
                'status': 'EXPECTED_FAIL' if 'too few' in str(e).lower() or 'empty' in str(e).lower() else 'FAIL',
                'error': str(e)
            }
            print(f"⚠️  EXPECTED/HANDLED: Very small file")
            print(f"   Error: {str(e)}")
    
    # Test with non-existent file
    print_test("Non-existent file test")
    
    try:
        from twisstntern.pipeline import run_analysis
        
        triangles_results, fundamental_results, csv_file_used = run_analysis(
            file="non_existent_file.csv",
            output_dir="test_output_nonexistent"
        )
        
        results['non_existent_file'] = {
            'status': 'UNEXPECTED_PASS',
            'note': 'Should have failed'
        }
        print(f"❌ UNEXPECTED PASS: Non-existent file should have failed")
        
    except Exception as e:
        results['non_existent_file'] = {
            'status': 'EXPECTED_FAIL',
            'error': str(e)
        }
        print(f"✅ EXPECTED FAIL: Non-existent file correctly rejected")
        print(f"   Error: {str(e)}")
    
    return results

def run_performance_tests():
    """Test performance with large files"""
    print_section("PERFORMANCE TESTING")
    
    results = {}
    
    # Test with large files
    large_files = [
        "csv files/gene_flow_sims/A_m0.csv",
        "csv files/gene_flow_sims/ne0.45.csv"
    ]
    
    for file_path in large_files:
        if not os.path.exists(file_path):
            continue
            
        print_test(f"Performance test - {Path(file_path).name}")
        
        try:
            from twisstntern.pipeline import run_analysis
            
            start_time = time.time()
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=file_path,
                output_dir=f"test_output_performance_{Path(file_path).stem}"
            )
            
            duration = time.time() - start_time
            
            # Check file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            results[f"performance_{Path(file_path).stem}"] = {
                'status': 'PASS',
                'duration': duration,
                'file_size_mb': file_size,
                'triangles_shape': triangles_results.shape,
                'performance_mb_per_sec': file_size / duration if duration > 0 else 0
            }
            
            print(f"✅ PASS: {Path(file_path).name}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   File size: {file_size:.1f} MB")
            print(f"   Performance: {file_size/duration:.1f} MB/s")
            print(f"   Triangles shape: {triangles_results.shape}")
            
        except Exception as e:
            results[f"performance_{Path(file_path).stem}"] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ FAIL: {Path(file_path).name}")
            print(f"   Error: {str(e)}")
    
    return results

def run_data_validation_tests():
    """Test data validation and consistency"""
    print_section("DATA VALIDATION TESTING")
    
    results = {}
    
    # Test with known good files and validate the results make sense
    test_files = [
        "csv files/gene_flow_sims/A_m0.csv",  # No migration
        "csv files/gene_flow_sims/B_m0.001.csv",  # Low migration
        "csv files/gene_flow_sims/F_m0.035.csv",  # Higher migration
    ]
    
    analysis_results = {}
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            continue
            
        print_test(f"Data validation - {Path(file_path).name}")
        
        try:
            from twisstntern.pipeline import run_analysis
            
            triangles_results, fundamental_results, csv_file_used = run_analysis(
                file=file_path,
                output_dir=f"test_output_validation_{Path(file_path).stem}"
            )
            
            # Validate the data makes sense
            n_right, n_left, d_lr, g_test, p_value = fundamental_results
            
            # Basic sanity checks
            assert n_right >= 0, "n_right should be non-negative"
            assert n_left >= 0, "n_left should be non-negative"
            assert n_right + n_left > 0, "Total count should be positive"
            assert -1 <= d_lr <= 1, "D_LR should be between -1 and 1"
            assert g_test >= 0, "G-test should be non-negative"
            assert 0 <= p_value <= 1, "p-value should be between 0 and 1"
            
            # Check triangles_results structure
            assert len(triangles_results) > 0, "Should have triangle results"
            required_columns = ['n_right', 'n_left', 'D_LR', 'G-test', 'p-value(g-test)']
            for col in required_columns:
                assert col in triangles_results.columns, f"Missing column: {col}"
            
            analysis_results[file_path] = {
                'n_right': n_right,
                'n_left': n_left,
                'd_lr': d_lr,
                'p_value': p_value,
                'total_triangles': len(triangles_results)
            }
            
            results[f"validation_{Path(file_path).stem}"] = {
                'status': 'PASS',
                'fundamental_results': fundamental_results,
                'triangles_count': len(triangles_results)
            }
            
            print(f"✅ PASS: {Path(file_path).name}")
            print(f"   n_right: {n_right}, n_left: {n_left}")
            print(f"   D_LR: {d_lr:.4f}, p-value: {p_value:.2e}")
            print(f"   Triangles analyzed: {len(triangles_results)}")
            
        except Exception as e:
            results[f"validation_{Path(file_path).stem}"] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ FAIL: {Path(file_path).name}")
            print(f"   Error: {str(e)}")
    
    # Cross-validation: compare results between files
    if len(analysis_results) >= 2:
        print_test("Cross-validation between datasets")
        
        files = list(analysis_results.keys())
        for i in range(len(files)):
            for j in range(i+1, len(files)):
                file1, file2 = files[i], files[j]
                result1, result2 = analysis_results[file1], analysis_results[file2]
                
                print(f"   Comparing {Path(file1).name} vs {Path(file2).name}:")
                print(f"     D_LR: {result1['d_lr']:.4f} vs {result2['d_lr']:.4f}")
                print(f"     p-value: {result1['p_value']:.2e} vs {result2['p_value']:.2e}")
    
    return results

def cleanup_test_directories():
    """Clean up all test output directories"""
    print_section("CLEANUP")
    
    test_dirs = glob.glob("test_output_*")
    cleaned = 0
    
    for test_dir in test_dirs:
        try:
            shutil.rmtree(test_dir)
            cleaned += 1
        except Exception as e:
            print(f"⚠️  Could not remove {test_dir}: {e}")
    
    print(f"🧹 Cleaned up {cleaned} test directories")

def generate_test_report(all_results):
    """Generate a comprehensive test report"""
    print_section("TEST REPORT")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    expected_fails = 0
    
    for test_category, results in all_results.items():
        print(f"\n📊 {test_category.upper()} RESULTS:")
        print("-" * 40)
        
        for test_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            total_tests += 1
            
            if status == 'PASS':
                passed_tests += 1
                print(f"   ✅ {test_name}: PASS")
            elif status == 'EXPECTED_FAIL':
                expected_fails += 1
                print(f"   ⚠️  {test_name}: EXPECTED FAIL")
            else:
                failed_tests += 1
                print(f"   ❌ {test_name}: FAIL")
                if 'error' in result:
                    print(f"      Error: {result['error']}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Expected failures: {expected_fails}")
    print(f"Success rate: {(passed_tests + expected_fails) / total_tests * 100:.1f}%")
    
    if failed_tests == 0:
        print(f"\n🎉 ALL TESTS PASSED! TWISSTNTERN is working perfectly!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Review the errors above.")
    
    return {
        'total': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'expected_fails': expected_fails,
        'success_rate': (passed_tests + expected_fails) / total_tests * 100
    }

def main():
    """Run the comprehensive test suite"""
    print("🧪 TWISSTNTERN Comprehensive Testing Suite")
    print("==========================================")
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Store all results
    all_results = {}
    
    try:
        # Run all test categories
        all_results['basic_functionality'] = run_basic_functionality_tests()
        all_results['parameter_testing'] = run_parameter_tests()
        all_results['edge_cases'] = run_edge_case_tests()
        all_results['performance'] = run_performance_tests()
        all_results['data_validation'] = run_data_validation_tests()
        
        # Generate final report
        summary = generate_test_report(all_results)
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing suite crashed: {e}")
        traceback.print_exc()
    finally:
        # Always cleanup
        cleanup_test_directories()
    
    print(f"\nTesting completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 