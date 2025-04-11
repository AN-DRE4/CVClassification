# Job Title Role Level Extraction

This script extracts job titles from resume data and categorizes them by role level (e.g., "entry_level", "mid_level", "senior_level", etc.).

## Description

The script processes a JSON file containing resume data, extracts job titles from the work experience sections, and categorizes them by role level. It can run in two modes:

1. **Automatic Mode:** The script uses predefined role level mappings to categorize job titles automatically.
2. **Interactive Mode:** The script allows human intervention to categorize job titles that haven't been categorized before.

## Features

- Extracts job titles from resumes' work experience sections
- Automatically categorizes titles based on a hierarchical role level structure
- Consolidates similar role keywords into standardized categories
- Provides an interactive mode for human categorization of new titles
- Saves categorized titles to a JSON file with role levels as keys and titles as values
- Incremental processing: already categorized titles are not processed again in interactive mode

## Usage

### Basic Usage

```bash
python extract_job_titles.py
```

This will use the default input file (`silver_labeled_resumes.json`) and output file (`job_titles_by_role_level.json`).

### Custom Input/Output Files

```bash
python extract_job_titles.py --input your_input_file.json --output your_output_file.json
```

### Interactive Mode

```bash
python extract_job_titles.py --interactive
```

## Output Format

The output JSON file has the following structure:

```json
{
  "entry_level": [
    "Junior Software Engineer",
    "Intern",
    "..."
  ],
  "senior_level": [
    "Senior Software Engineer",
    "Lead Developer",
    "..."
  ],
  "management": [
    "Project Manager",
    "Director of Engineering",
    "..."
  ],
  "uncategorized": [
    "Software Engineer",
    "Data Scientist",
    "..."
  ]
}
```

## Role Level Categories

The script uses the following role level categories with their associated keywords:

- **entry_level**: 
  - entry, junior, jr, intern, trainee, fresher, graduate

- **mid_level**: 
  - associate, staff, analyst, consultant, specialist

- **senior_level**: 
  - senior, sr, lead, principal, tech lead, technical lead, expert

- **management**: 
  - manager, director, vp, chief, head, executive, officer

If no match is found, the title is categorized as "uncategorized".

## Requirements

- Python 3.6+
- Standard libraries: json, re, os, collections, argparse 