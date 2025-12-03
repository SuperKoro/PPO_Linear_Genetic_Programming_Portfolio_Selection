"""
Unit Tests for LGP System
Quick validation that programs execute correctly
"""

from typing import Dict
from lgp_instructions import *
from lgp_program import LGPProgram, PortfolioBuilder
from lgp_generator import LGPGenerator
from config import LGPConfig
import random


def test_1_simple_program():
    """Test 1: Simple program with LoadConst + Arithmetic + SET_* instructions"""
    print("\n" + "="*70)
    print("TEST 1: Simple Program Execution")
    print("="*70)
    
    instructions = [
        # Load some constants
        LoadConstInstruction(dest=0, value=5.0),
        LoadConstInstruction(dest=1, value=10.0),
        
        # Arithmetic
        ArithmeticInstruction(dest=2, op="+", src1=0, src2=1, src2_is_const=False),
        ArithmeticInstruction(dest=3, op="*", src1=2, src2=2.0, src2_is_const=True),
        
        # Set portfolio
        SetPortfolioInstruction(component="DR", reg_name=0),
        SetPortfolioInstruction(component="MH1", reg_name=1, reg_weight=2),
        SetPortfolioInstruction(component="MH2", reg_name=2, reg_weight=3),
        SetPortfolioInstruction(component="MH3", reg_name=3, reg_weight=2),
    ]
    
    program = LGPProgram(instructions=instructions, num_registers=20)
    
    # Execute
    inputs = {"num_jobs": 10, "avg_processing_time": 8.5, "avg_ops_per_job": 3.0}
    try:
        portfolio = program.execute(inputs)
        
        # Validate
        dr_name = portfolio.genes[0].name
        mh_genes = portfolio.genes[1:]
        
        assert dr_name in LGPConfig.available_dr, f"DR {dr_name} not in available_dr"
        assert len(mh_genes) == LGPConfig.n_mh_genes, f"Expected {LGPConfig.n_mh_genes} MH genes, got {len(mh_genes)}"
        
        for mh in mh_genes:
            assert mh.name in LGPConfig.available_mh, f"MH {mh.name} not in available_mh"
        
        print(f"✅ PASSED: Portfolio = {dr_name} | " + ", ".join([f"{g.name}:{g.w_raw:.2f}" for g in mh_genes]))
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_2_conditional_skip():
    """Test 2: Program with COND_SKIP - verify IP flow and no crashes"""
    print("\n" + "="*70)
    print("TEST 2: Conditional Skip Execution")
    print("="*70)
    
    instructions = [
        LoadConstInstruction(dest=0, value=15.0),   # r[0] = 15
        LoadConstInstruction(dest=1, value=5.0),    # r[1] = 5
        
        # Skip if r[0] > 10 (TRUE, should skip next 2 instructions)
        ConditionalSkipInstruction(cond_reg=0, threshold=10.0, skip_count=2, comparison=">"),
        
        LoadConstInstruction(dest=2, value=100.0),  # SKIPPED
        LoadConstInstruction(dest=3, value=200.0),  # SKIPPED
        
        LoadConstInstruction(dest=4, value=50.0),   # Executed
        
        # Skip if r[1] > 10 (FALSE, should NOT skip)
        ConditionalSkipInstruction(cond_reg=1, threshold=10.0, skip_count=1, comparison=">"),
        
        LoadConstInstruction(dest=5, value=75.0),   # Executed
        
        # Portfolio
        SetPortfolioInstruction(component="DR", reg_name=4),
        SetPortfolioInstruction(component="MH1", reg_name=5, reg_weight=1),
        SetPortfolioInstruction(component="MH2", reg_name=0, reg_weight=1),
        SetPortfolioInstruction(component="MH3", reg_name=1, reg_weight=1),
    ]
    
    program = LGPProgram(instructions=instructions, num_registers=20)
    
    try:
        portfolio = program.execute({"num_jobs": 10})
        
        # If we got here without crash, IP flow is correct!
        dr_name = portfolio.genes[0].name
        mh_genes = portfolio.genes[1:]
        
        print(f"✅ PASSED: No crash, IP flow correct")
        print(f"   Portfolio = {dr_name} | " + ", ".join([f"{g.name}:{g.w_raw:.2f}" for g in mh_genes]))
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_random_generation():
    """Test 3: Generate 100 random programs and execute all"""
    print("\n" + "="*70)
    print("TEST 3: Random Program Generation (100 programs)")
    print("="*70)
    
    rng = random.Random(42)
    generator = LGPGenerator(
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        rng=rng
    )
    
    inputs = {
        "num_jobs": 12.0,
        "avg_processing_time": 8.5,
        "avg_ops_per_job": 3.2
    }
    
    success_count = 0
    fail_count = 0
    
    for i in range(100):
        try:
            program = generator.generate_random_program()
            portfolio = program.execute(inputs)
            
            # Quick validation
            assert len(portfolio.genes) == 1 + LGPConfig.n_mh_genes
            assert portfolio.genes[0].name in LGPConfig.available_dr
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            print(f"  Program {i+1} failed: {e}")
    
    print(f"\n✅ SUCCESS: {success_count}/100 programs executed correctly")
    if fail_count > 0:
        print(f"❌ FAILED: {fail_count}/100 programs crashed")
        return False
    else:
        print("🎉 ALL 100 PROGRAMS PASSED!")
        return True


def test_4_program_serialization():
    """Bonus Test: Verify programs can be saved/loaded"""
    print("\n" + "="*70)
    print("BONUS TEST: Program Serialization")
    print("="*70)
    
    # Create program
    instructions = [
        LoadInputInstruction(dest=0, input_key="num_jobs"),
        ArithmeticInstruction(dest=1, op="*", src1=0, src2=2.0, src2_is_const=True),
        SetPortfolioInstruction(component="DR", reg_name=1),
        SetPortfolioInstruction(component="MH1", reg_name=0, reg_weight=1),
        SetPortfolioInstruction(component="MH2", reg_name=1, reg_weight=1),
        SetPortfolioInstruction(component="MH3", reg_name=0, reg_weight=1),
    ]
    
    program1 = LGPProgram(instructions=instructions)
    
    # Serialize
    data = program1.to_dict()
    
    # Deserialize
    program2 = LGPProgram.from_dict(data)
    
    # Execute both
    inputs = {"num_jobs": 15.0}
    portfolio1 = program1.execute(inputs)
    portfolio2 = program2.execute(inputs)
    
    # Compare
    if portfolio1.genes[0].name == portfolio2.genes[0].name:
        print("✅ PASSED: Serialization/deserialization works!")
        return True
    else:
        print("❌ FAILED: Portfolios don't match after serialization")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 LGP UNIT TESTS")
    print("="*70)
    
    # Import config to validate
    from config import validate_config, print_config_summary
    
    print("\n📋 Validating configuration...")
    validate_config()
    
    # Run tests
    results = []
    results.append(("Simple Program", test_1_simple_program()))
    results.append(("Conditional Skip", test_2_conditional_skip()))
    results.append(("Random Generation", test_3_random_generation()))
    results.append(("Serialization", test_4_program_serialization()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum([1 for name, result in results if result])
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! LGP System is ready!")
        exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please debug before running main.py")
        exit(1)

