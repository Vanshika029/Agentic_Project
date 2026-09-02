import pytest

def test_skill_gap_set_subtraction():
    required_skills = {'Python', 'SQL', 'MLOps', 'Docker', 'AWS'}
    employee_has = {'Python', 'SQL', 'AWS'}
    
    gap = required_skills - employee_has
    assert gap == {'MLOps', 'Docker'}

def test_skill_gap_severity_rules():
    def get_severity(count):
        if count >= 100:
            return 'HIGH'
        elif count >= 50:
            return 'MEDIUM'
        return 'LOW'
        
    assert get_severity(120) == 'HIGH'
    assert get_severity(98) == 'MEDIUM'
    assert get_severity(50) == 'MEDIUM'
    assert get_severity(30) == 'LOW'
