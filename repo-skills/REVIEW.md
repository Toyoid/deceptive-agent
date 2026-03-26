# Coding Agent Self-Review Guidelines

This is the final review pass before handing the code back to the human developer. Focus strictly on the correctness, completeness, consistency, elegance, and cleanliness of the implementation.

## Core Philosophy
- **Active Development**: This repository is moving fast. We value clear, concise, and clean code over hyper-defensive, bloated implementations.
- **No Scope Creep**: Do NOT add new features, refactor unrelated modules, or "fix" things outside the explicit scope of the task.
- **Lean & Elegant**: Avoid overly conservative coding. Do not add unnecessary safeguarding code (excessive try-catches, redundant null-checks). Trust the existing architecture. 

## Review Checklist

### 1. Correctness & Completeness
- [ ] Have all explicit requirements of the requested feature been met?
- [ ] Does the implementation handle the primary logic elegantly?
- [ ] Is the code free of logical flaws, syntax errors, and type mismatches?

### 2. Cleanliness & Elegance
- [ ] Is the code easy to read and mentally parse? 
- [ ] Are variables and functions named clearly, intentionally, and concisely?
- [ ] Can any complex, nested logic be simplified or extracted?

### 3. Consistency
- [ ] Does the new code tightly match the existing style, formatting, and design patterns in the repository?
- [ ] Are we reusing existing utilities and abstractions rather than reinventing the wheel?

### 4. What NOT to do
- [ ] **NO** adding "nice-to-have" features that weren't requested.
- [ ] **NO** wrapping everything in generic try-catch blocks just to be "safe".
- [ ] **NO** over-engineering or premature optimization. 

**Goal**: Hand back a tightly scoped, clean, consistent, and fully operational implementation.
