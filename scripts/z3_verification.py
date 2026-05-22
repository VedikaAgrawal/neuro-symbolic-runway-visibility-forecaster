import os
import sys
import z3

class SymbolicGuardrail:
    """
    M.Tech Thesis Neuro-Symbolic Guardrail Layer.
    Intercepts connectionist visibility predictions at runtime and cross-examines
    them against deterministic meteorological axioms using Microsoft Research's Z3 SMT Solver.
    """
    def __init__(self):
        # Initialize Z3 Solver instance
        self.solver = z3.Solver()
        
        # Declare symbolic real numbers for predictor features and target visibility
        self.visibility = z3.Real('visibility')
        self.rh = z3.Real('rh')
        self.dpd = z3.Real('dpd')
        self.wsi = z3.Real('wsi')
        self.aod = z3.Real('aod')
        
        # Axiom 1: Dry Air Fog Impossibility Rule
        # A visibility collapse (< 800m) is physically impossible in very dry air (RH < 45% or DPD > 12°C)
        self.axiom_dry_air = z3.Implies(
            z3.Or(self.rh < 45.0, self.dpd > 12.0),
            self.visibility >= 800.0
        )
        
        # Axiom 2: Saturated Air Stagnation Limit
        # High visibility (> 2500m) is physically impossible in saturated, stagnant air (RH >= 98% and WSI == 1.0)
        self.axiom_stagnation = z3.Implies(
            z3.And(self.rh >= 98.0, self.wsi == 1.0),
            self.visibility <= 2500.0
        )
        
        # Axiom 3: Aerosol Attenuation Boundary
        # Maximum optical range (> 3500m) is physically impossible under extreme Kanpur aerosol optical load (AOD > 1.8)
        self.axiom_aerosol = z3.Implies(
            self.aod > 1.8,
            self.visibility <= 3500.0
        )
        
        # Permanently assert meteorological laws in the solver workspace
        self.solver.add(self.axiom_dry_air)
        self.solver.add(self.axiom_stagnation)
        self.solver.add(self.axiom_aerosol)

    def verify_prediction(self, raw_pred, rh_val, dpd_val, wsi_val, aod_val):
        """
        Audits a single multi-horizon forecast step.
        Returns: (is_sat: bool, corrected_val: float, status: str, violated_rules: list)
        """
        import math
        
        # Check for NaN values and handle them gracefully
        if (math.isnan(raw_pred) or math.isnan(rh_val) or 
            math.isnan(dpd_val) or math.isnan(wsi_val) or math.isnan(aod_val)):
            return True, raw_pred, "SAT", []
            
        # Ensure values are float native types
        raw_pred = float(raw_pred)
        rh_val = float(rh_val)
        dpd_val = float(dpd_val)
        wsi_val = float(wsi_val)
        aod_val = float(aod_val)
        
        # Push solver state to create a temporary scope
        self.solver.push()
        
        # Bind the specific situational input parameters
        self.solver.add(self.rh == rh_val)
        self.solver.add(self.dpd == dpd_val)
        self.solver.add(self.wsi == wsi_val)
        self.solver.add(self.aod == aod_val)
        
        # Bind the incoming black-box prediction
        self.solver.add(self.visibility == raw_pred)
        
        # Perform satisfiability check
        check_res = self.solver.check()
        
        # Discard situational constraints from solver workspace
        self.solver.pop()
        
        violated_rules = []
        corrected_pred = raw_pred
        
        if check_res == z3.sat:
            return True, raw_pred, "SAT", []
        else:
            # Audit which physical axioms were violated and compute physics-safe corrected fallbacks
            
            # Check Rule 1: Dry Air Violation
            if (rh_val < 45.0 or dpd_val > 12.0) and raw_pred < 800.0:
                violated_rules.append("Rule 1: Dry Air Fog Impossibility")
                corrected_pred = max(corrected_pred, 800.0)
                
            # Check Rule 2: Stagnation Violation
            if (rh_val >= 98.0 and wsi_val == 1.0) and raw_pred > 2500.0:
                violated_rules.append("Rule 2: Saturated Air Stagnation Limit")
                corrected_pred = min(corrected_pred, 2500.0)
                
            # Check Rule 3: Aerosol Violation
            if aod_val > 1.8 and raw_pred > 3500.0:
                violated_rules.append("Rule 3: Aerosol Attenuation Boundary")
                corrected_pred = min(corrected_pred, 3500.0)
                
            return False, corrected_pred, "UNSAT", violated_rules

def run_tests():
    print("🔬 Running Z3 meteorological symbolic guardrail test suite...")
    guard = SymbolicGuardrail()
    
    # Test 1: Physically valid state
    is_sat, val, status, rules = guard.verify_prediction(
        raw_pred=1500.0, rh_val=60.0, dpd_val=5.0, wsi_val=0.0, aod_val=0.5
    )
    print(f"Test 1 (Valid State): is_sat={is_sat} | val={val}m | status={status} | rules={rules}")
    assert is_sat == True
    
    # Test 2: Dry Air Violation (Model predicted 400m fog in 30% RH)
    is_sat, val, status, rules = guard.verify_prediction(
        raw_pred=400.0, rh_val=30.0, dpd_val=15.0, wsi_val=0.0, aod_val=0.4
    )
    print(f"Test 2 (Dry Air UNSAT): is_sat={is_sat} | val={val}m | status={status} | rules={rules}")
    assert is_sat == False
    assert val == 800.0
    
    # Test 3: Stagnation Violation (Model predicted 4000m clear visibility in saturated stagnation)
    is_sat, val, status, rules = guard.verify_prediction(
        raw_pred=4000.0, rh_val=99.0, dpd_val=0.2, wsi_val=1.0, aod_val=0.5
    )
    print(f"Test 3 (Stagnation UNSAT): is_sat={is_sat} | val={val}m | status={status} | rules={rules}")
    assert is_sat == False
    assert val == 2500.0
    
    # Test 4: Aerosol Violation (Model predicted 4500m visibility under extreme Kanpur AOD load)
    is_sat, val, status, rules = guard.verify_prediction(
        raw_pred=4500.0, rh_val=40.0, dpd_val=8.0, wsi_val=0.0, aod_val=2.2
    )
    print(f"Test 4 (Aerosol UNSAT): is_sat={is_sat} | val={val}m | status={status} | rules={rules}")
    assert is_sat == False
    assert val == 3500.0
    
    print("\n✅ All Z3 meteorological solver test constraints successfully satisfied!")

if __name__ == "__main__":
    run_tests()
