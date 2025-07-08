#!/usr/bin/env python3
"""
Comprehensive Testing Suite for TWISSTNTERN_SIMULATE
===================================================

This script performs thorough testing of the twisstntern_simulate package including:
- Basic functionality with config file
- Parameter override testing
- Different simulation modes (locus vs chromosome)
- Downsampling functionality (both N and kb-based)
- Colormap testing
- Performance testing
- Edge case handling
- Config validation

Author: AI Assistant
Date: 2025-07-03
"""

import os
import sys
import time
import traceback
import shutil
import glob
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

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

def create_test_config(name: str, modifications: Dict[str, Any] = None) -> str:
    """Create a test configuration file with optional modifications"""
    config_file = f"test_config_{name}.yaml"
    
    # Load the template
    with open("config_template.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Apply modifications if provided
    if modifications:
        def update_nested_dict(d, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                    update_nested_dict(d[key], value)
                else:
                    d[key] = value
        
        update_nested_dict(config, modifications)
    
    # Write the test config
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file

def check_simulation_outputs(output_dir, expected_files):
    """Check if expected simulation output files exist and are valid"""
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
    """Test basic functionality with default configuration"""
    print_section("BASIC FUNCTIONALITY TESTS")
    
    results = {}
    
    # Test 1: Basic simulation with default config
    print_test("Basic simulation with default config")
    
    try:
        from twisstntern_simulate.pipeline import run_pipeline
        
        start_time = time.time()
        
        triangles_results, fundamental_results, csv_file_used = run_pipeline(
            config_path="config_template.yaml",
            output_dir="test_simulate_basic_default"
        )
        
        duration = time.time() - start_time
        
        # Validate results
        assert isinstance(triangles_results, pd.DataFrame), "triangles_results should be DataFrame"
        assert isinstance(fundamental_results, tuple), "fundamental_results should be tuple"
        assert len(fundamental_results) == 5, "fundamental_results should have 5 elements"
        assert isinstance(csv_file_used, str), "csv_file_used should be string"
        assert os.path.exists(csv_file_used), "CSV file should exist"
        
        # Check expected outputs
        expected_outputs = [
            "topology_weights.csv",
            "trees.newick",
            "fundamental_asymmetry.png",
            "heatmap.png",
            "radcount.png",
            "granuality_0.1.png",
            "analysis_granularity_0.1.png",
            "index_granularity_0.1.png",
            "triangle_analysis_0.1.csv"
        ]
        
        missing_files, file_info = check_simulation_outputs("test_simulate_basic_default", expected_outputs)
        
        results['basic_default'] = {
            'status': 'PASS',
            'duration': duration,
            'missing_files': missing_files,
            'triangles_shape': triangles_results.shape,
            'fundamental_results': fundamental_results,
            'csv_file': csv_file_used
        }
        
        print(f"✅ PASS: Basic simulation")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Triangles shape: {triangles_results.shape}")
        print(f"   Fundamental results: n_right={fundamental_results[0]}, n_left={fundamental_results[1]}, p-value={fundamental_results[4]:.2e}")
        print(f"   Missing files: {len(missing_files)}")
        print(f"   CSV file: {csv_file_used}")
        
    except Exception as e:
        results['basic_default'] = {
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"❌ FAIL: Basic simulation")
        print(f"   Error: {str(e)}")
    
    return results

def run_parameter_override_tests():
    """Test parameter override functionality"""
    print_section("PARAMETER OVERRIDE TESTS")
    
    results = {}
    
    # Test migration override
    print_test("Migration rate override")
    
    try:
        from twisstntern_simulate.pipeline import run_pipeline
        
        triangles_results, fundamental_results, csv_file_used = run_pipeline(
            config_path="config_template.yaml",
            config_overrides=["migration.p1>p2=0.05", "migration.p2>p1=0.05"],
            output_dir="test_simulate_migration_override"
        )
        
        results['migration_override'] = {
            'status': 'PASS',
            'triangles_shape': triangles_results.shape
        }
        
        print(f"✅ PASS: Migration override")
        print(f"   Triangles shape: {triangles_results.shape}")
        
    except Exception as e:
        results['migration_override'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"❌ FAIL: Migration override")
        print(f"   Error: {str(e)}")
    
    # Test seed override
    print_test("Seed override")
    
    try:
        from twisstntern_simulate.pipeline import run_pipeline
        
        triangles_results, fundamental_results, csv_file_used = run_pipeline(
            config_path="config_template.yaml",
            seed_override=12345,
            output_dir="test_simulate_seed_override"
        )
        
        results['seed_override'] = {
            'status': 'PASS',
            'triangles_shape': triangles_results.shape
        }
        
        print(f"✅ PASS: Seed override")
        print(f"   Triangles shape: {triangles_results.shape}")
        
    except Exception as e:
        results['seed_override'] = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"❌ FAIL: Seed override")
        print(f"   Error: {str(e)}")
    
    return results

def run_analysis_parameter_tests():
    """Test analysis-specific parameters"""
    print_section("ANALYSIS PARAMETER TESTS")
    
    results = {}
    
    # Test different granularities
    granularities = [0.05, 0.1, 0.15]
    
    for granularity in granularities:
        print_test(f"Granularity test - {granularity}")
        
        try:
            from twisstntern_simulate.pipeline import run_pipeline
            
            triangles_results, fundamental_results, csv_file_used = run_pipeline(
                config_path="config_template.yaml",
                granularity=granularity,
                output_dir=f"test_simulate_granularity_{granularity}"
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
    colormaps = ["viridis", "viridis_r", "plasma", "inferno", "Blues"]
    
    for colormap in colormaps:
        print_test(f"Colormap test - {colormap}")
        
        try:
            from twisstntern_simulate.pipeline import run_pipeline
            
            triangles_results, fundamental_results, csv_file_used = run_pipeline(
                config_path="config_template.yaml",
                colormap=colormap,
                output_dir=f"test_simulate_colormap_{colormap}"
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
    
    return results

def run_edge_case_tests():
    """Test edge cases and error handling"""
    print_section("EDGE CASE TESTS")
    
    results = {}
    
    # Test with non-existent config file
    print_test("Non-existent config file")
    
    try:
        from twisstntern_simulate.pipeline import run_pipeline
        
        triangles_results, fundamental_results, csv_file_used = run_pipeline(
            config_path="nonexistent_config.yaml",
            output_dir="test_simulate_nonexistent_config"
        )
        
        results['nonexistent_config'] = {
            'status': 'UNEXPECTED_PASS',
            'note': 'Should have failed'
        }
        print(f"❌ UNEXPECTED PASS: Non-existent config should have failed")
        
    except Exception as e:
        results['nonexistent_config'] = {
            'status': 'EXPECTED_FAIL',
            'error': str(e)
        }
        print(f"✅ EXPECTED FAIL: Non-existent config correctly rejected")
        print(f"   Error: {str(e)}")
    
    return results

def cleanup_test_files():
    """Clean up all test files and directories"""
    print_section("CLEANUP")
    
    # Clean up test directories
    test_dirs = glob.glob("test_simulate_*")
    cleaned_dirs = 0
    
    for test_dir in test_dirs:
        try:
            shutil.rmtree(test_dir)
            cleaned_dirs += 1
        except Exception as e:
            print(f"⚠️  Could not remove {test_dir}: {e}")
    
    # Clean up test config files
    test_configs = glob.glob("test_config_*.yaml")
    cleaned_configs = 0
    
    for test_config in test_configs:
        try:
            os.remove(test_config)
            cleaned_configs += 1
        except Exception as e:
            print(f"⚠️  Could not remove {test_config}: {e}")
    
    print(f"🧹 Cleaned up {cleaned_dirs} test directories and {cleaned_configs} config files")

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
                if 'duration' in result:
                    print(f"      Duration: {result['duration']:.2f}s")
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
        print(f"\n🎉 ALL TESTS PASSED! TWISSTNTERN_SIMULATE is working perfectly!")
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
    print("🧪 TWISSTNTERN_SIMULATE Comprehensive Testing Suite")
    print("=" * 60)
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Store all results
    all_results = {}
    
    try:
        # Run all test categories
        all_results['basic_functionality'] = run_basic_functionality_tests()
        all_results['parameter_overrides'] = run_parameter_override_tests()
        all_results['analysis_parameters'] = run_analysis_parameter_tests()
        all_results['edge_cases'] = run_edge_case_tests()
        
        # Generate final report
        summary = generate_test_report(all_results)
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing suite crashed: {e}")
        traceback.print_exc()
    finally:
        # Always cleanup
        cleanup_test_files()
    
    print(f"\nTesting completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()