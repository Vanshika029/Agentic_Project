# Enterprise HR AI — Data Relationships & Entity Architecture

This document formalizes the entity relationships, join keys, and schemas connecting the five core datasets in the platform.

```
                    ┌─────────────────────────┐
                    │      EMPLOYEE MASTER    │
                    │  (employee_attrition)   │
                    │   PK: EmployeeID        │
                    └────────────┬────────────┘
                                 │
                 1 : 1           │           * : 1
          ┌──────────────────────┴──────────────────────┐
          │                                             │
          ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│       ENGAGEMENT        │                   │    OCCUPATION MASTER    │
│(hr_performance_engagem.)│                   │   (occupation_master)   │
│     FK: EmployeeID      │                   │     PK: ONET_SOC_Code   │
└─────────────────────────┘                   └────────────┬────────────┘
                                                           │
                                           1 : *           │           1 : *
                                    ┌──────────────────────┴──────────────────────┐
                                    │                                             │
                                    ▼                                             ▼
                          ┌───────────────────┐                         ┌───────────────────┐
                          │ ESSENTIAL SKILLS  │                         │  SOFTWARE SKILLS  │
                          │(essential_skills) │                         │ (software_skills) │
                          │ FK: ONET_SOC_Code │                         │ FK: ONET_SOC_Code │
                          └───────────────────┘                         └───────────────────┘
```

## Relational Join Specification

| Source Table | Target Table | Source Key | Target Key | Cardinality | Business Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `employee_attrition` | `hr_performance_engagement` | `EmployeeID` | `EmployeeID` | 1 : 1 | Connects demographic/attrition features with performance ratings and engagement metrics. |
| `employee_attrition` | `occupation_master` | `JobRole` | `Title` / `ONET_Code` | * : 1 | Maps company-specific job titles to O*NET standard occupational competencies. |
| `occupation_master` | `essential_skills` | `O*NET-SOC Code` | `O*NET-SOC Code` | 1 : * | Retrieves core behavioral and technical cognitive skills required per occupation. |
| `occupation_master` | `software_skills` | `O*NET-SOC Code` | `O*NET-SOC Code` | 1 : * | Retrieves software tools, programming languages, and tech stacks required per occupation. |
| `employee_attrition` | `employee_skills_inventory` | `EmployeeID` | `EmployeeID` | 1 : * | Links individual employee profiles to their verified proficiencies for skill gap calculation. |

## Data Schemas & Constraints

1. **Employee Master (`employee_attrition_processed.csv`)**:
   - `EmployeeID`: Unique Integer identifier [1..N]
   - `Age`: Integer in [18, 100]
   - `Attrition`: Categorical string in {'Yes', 'No'}
   - `MonthlyIncome`: Numeric (> 0)
   - `JobRole`: Categorical string (9 enterprise roles)
   - `OverTime`: Categorical string in {'Yes', 'No'}

2. **Engagement (`engagement_processed.csv`)**:
   - `EmployeeID`: Foreign key to Employee Master
   - `EngagementScore`: Integer in [0, 100]
   - `PerformanceRating`: Integer in [1, 5]
   - `WorkLifeBalanceScore`: Integer in [1, 5]

3. **Occupation Competency (`occupation_master.csv`, `essential_skills_processed.csv`, `software_skills_processed.csv`)**:
   - `O*NET-SOC Code`: Unique occupational alphanumeric identifier (e.g. `15-1221.00`)
   - `Element Name`: Standard skill competency name
   - `Workplace Example`: Specific software application/tool name (e.g. `Python`, `AWS`, `Docker`)
