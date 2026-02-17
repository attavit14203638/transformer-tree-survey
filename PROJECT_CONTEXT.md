# Project Context: Literature Review GitHub Repository

**Date Created**: January 2025  
**Last Updated**: February 2026  
**Purpose**: Context documentation for future development sessions

## 📋 Project Overview

### **Objective**
Create a GitHub repository to serve as **supplementary material** for a systematic literature review paper on transformer-based approaches for tree extraction. This is NOT an "awesome list" for community contribution, but rather a professional academic resource index.

### **Main Literature Review Project**
- **Paper Title**: Transformer-Based Tree Extraction from Remote Sensing Imagery: A Comprehensive Review
- **Authors**: Attavit Wilaiwongsakul, Bin Liang, Bryan Zheng, Fang Chen
- **Coverage**: 2017-2025
- **Total Papers**: 63 studies analyzed
- **Main Files**: 
  - `68f823600ac5436c4d362b39/main.tex` - LaTeX manuscript
  - `68f823600ac5436c4d362b39/bib/` - Bibliography files
  - `GitHub_Release/` - Repository for publication

## 🎯 Repository Purpose & Design Philosophy

### **Core Function**
- **Resource Discovery Hub**: Quick access to papers, datasets, code
- **Supplementary Material**: Supporting the main literature review paper
- **NOT Duplicating Analysis**: Detailed analysis stays in the published paper

### **Target Audience**
- Researchers looking for relevant papers and resources
- People who read the literature review paper and want to access original sources
- NOT community contributors or awesome-list maintainers

### **Key Design Principles**
1. **Clean Resource Index**: Focus on "where to find" not "what we found"
2. **Academic Professional**: Serious research supplement, not community list
3. **Weekly Updates**: Author adds papers regularly during research phase
4. **No Analysis Duplication**: Performance comparisons and gaps analysis in paper only

## 📁 Repository Structure Implemented

```
GitHub_Release/
├── README.md                 # Main repository page
├── LICENSE                   # MIT License
├── .gitignore               # Standard ignores for research projects
├── templates/               # [FUTURE] Templates for paper additions
├── docs/                    # [FUTURE] Additional documentation
├── assets/                  # [FUTURE] Images and figures
├── papers/                  # [FUTURE] Organized paper storage
├── summaries/               # [FUTURE] Brief summaries
└── tools/                   # [FUTURE] Analysis tools
```

## 📚 README.md Structure Finalized

### **Sections Included**
1. **📄 Literature Review Paper** - Link to published paper
2. **📊 Survey Figures** - Key summary diagrams (10 figures)
3. **📝 Literature Overview** - Brief statistics (63 papers, 2017-2025)
4. **📈 Star History** - Repository star growth chart
5. **📚 Research Papers** - Chronological by architecture type
6. **📊 Datasets & Benchmarks** - Access-focused resource list
7. **🔧 Available Code & Tools** - Implementation links
8. **📑 Citation & Usage** - How to cite and use repository

### **Sections REMOVED** (from awesome-list inspiration)
- ❌ Awesome badge
- ❌ Highlights section
- ❌ Performance analysis (belongs in paper)
- ❌ Research gaps analysis (belongs in paper)
- ❌ Community contribution guidelines
- ❌ Getting started guides

## 🔧 Paper Entry Format Established

### **Format Convention**
```markdown
**[YYYY.MM] Paper Title**
- **Authors**: First Author, Second Author, et al.
- **Venue**: Journal/Conference Name
- **Key Contribution**: One-line summary of main contribution
- **Links**: 📖 [Paper](link) | 💻 [Code](link) | 📊 [Dataset](link)
```

### **Key Rules**
1. **Date Format**: `[2025.01]` for January, `[2025]` if month unknown
2. **Ordering**: Chronological within architecture categories (newest first)
3. **Link Consistency**: ALL paper links use `[Paper](link)` placeholder
4. **Link Omission**: Don't show unavailable resources (no "Not Available" text)
5. **One-line Contributions**: Brief, focused contribution summary

## 📊 Update Workflow

### **Process**
1. **Add new papers** → Update `GitHub_Release/README.md`
2. **Verify citations** → Check against `bib/Research_Papers.bib`
3. **Update statistics** → Maintain paper counts in README

## 🎨 Architecture Categories Defined

### **Paper Organization** (63 total)
1. **Pure Vision Transformers** (6 papers) - ViT-only approaches
2. **Hierarchical Vision Transformers** (8 papers) - Swin, Twins-SVT, SegFormer based
3. **CNN-Transformer Hybrids** (31 papers) - TransUNet variants, Swin-based, DETR-based
4. **Foundation Models** (12 papers) - SAM, SAM2, Prithvi, DOFA, FoMo-Net, etc.
5. **Vision-Language Models** (4 papers) - Tree-GPT, EarthDial, GeoLangBind, REO-VLM

## 🔄 Completion Status

### **Completed Tasks**
- [x] All 63 papers documented with metadata and links
- [x] Papers organized by architecture category
- [x] Paper links verified (DOI/arXiv where available)
- [x] Code repository links added where available
- [x] Dataset links included

### **Code/Dataset Integration**
- **Existing Code**: Links to original authors' implementations included
- **Dataset Links**: Access information provided for major datasets
- **FoMo-Bench**: Primary code and dataset resource for forest monitoring

## 💡 Key Decisions Made

### **Repository Philosophy**
- ✅ Academic supplementary material (not awesome list)
- ✅ Resource index focus (not analysis duplication)
- ✅ Professional appearance for publication
- ✅ Weekly manual updates (no automation needed)

### **Format Choices**
- ✅ Chronological ordering within categories
- ✅ Consistent link formatting across all papers
- ✅ Brief contribution summaries (detailed analysis in paper)
- ✅ Date tags for clear temporal organization

### **Maintenance Approach**
- ✅ Manual updates through Cursor IDE
- ✅ AI assistance for coordination across files
- ✅ Independent file structure (no cross-linking)
- ✅ Statistics tracking in taxonomy file

## 📝 Author Preferences Noted

1. **Word Choice**: Prefers not to use "fidelity" (noted in memories)
2. **Editor Setting**: Prefers word wrap on without column limit (noted in memories)
3. **Update Frequency**: Weekly basis during active research phase
4. **Link Consistency**: ALL papers use `[Paper](link)` regardless of type
5. **Manual Control**: Prefers manual updates over automation

## 🚀 Repository Complete

The repository is finalized for:
- **Publication**: Ready to be referenced in the systematic review paper
- **Resource Sharing**: 62 papers organized with direct access links
- **Professional Presentation**: Suitable for academic supplementary material

**Status**: ✅ Complete (February 2026)

### **Statistics**
- **Research Papers**: 63
- **Foundational Architectures**: 14
- **Related Surveys**: 15
- **Dataset Papers**: 10
- **Policy Papers**: 5
- **Total References**: ~107
- **Time Period**: 2017-2025
- **Geographic Coverage**: 27 countries across 6 continents
- **Figures Included**: 10 survey figures
