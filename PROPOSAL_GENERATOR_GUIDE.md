# Enhanced Research Proposal Generator - User Guide

## Overview

We now have **TWO tools** for generating research proposals:

### 1. **Original Tool**: `generate_research_proposals.py`
- Generates proposals from **predefined topics**
- Good for exploring specific research directions
- Takes 75-100 minutes for 5 proposals

### 2. **Enhanced Tool**: `generate_proposals_enhanced.py` ✨ **NEW!**
- Can use **local markdown documents** as inspiration
- Supports **both** document-based AND predefined topics
- Flexible CLI with multiple modes

---

## Quick Start

### Generate from Your Own Document

```bash
# Single document
python generate_proposals_enhanced.py \
  --mode document \
  --doc hierarchical_contextual_encoding_with_dynamic_attentional_pruning_*.md

# Multiple documents
python generate_proposals_enhanced.py \
  --mode document \
  --doc doc1.md \
  --doc doc2.md \
  --doc doc3.md
```

### Generate from Predefined Topics

```bash
python generate_proposals_enhanced.py --mode auto
```

### Do Both!

```bash
python generate_proposals_enhanced.py \
  --mode both \
  --doc your_research_idea.md
```

---

## What Just Happened with HiCEP-DAP?

✅ **Successfully Generated!**

We just converted your HiCEP-DAP research document into a comprehensive, multi-agent analyzed proposal!

**Input:** `hierarchical_contextual_encoding_with_dynamic_attentional_pruning_(hicep-dap)_for_ultra-efficient_llm_training_and_inference__deep_dive_and_refinement.md`

**Output:** `research/proposals/hicep_dap/proposal_from_hierarchical_contextual_encoding_with_dynamic_attentional_pruning__hicep-dap__for_ultra-efficient_ll.json`

**Analysis by 5 Agents:**
1. 🐍 **Python Engineer** - Implementation approach
2. 🧠 **AI Algorithms** - Technical challenges & innovations
3. 🏗️ **Systems Design** - Architecture & scalability
4. 🔄 **Training Pipeline** - Training strategy & experiments
5. 🎨 **Architecture Design** - Component design & integration

---

## File Structure

```
generate_research_proposals.py          # Original tool (predefined topics)
generate_proposals_enhanced.py          # New tool (documents + topics) ✨

research/proposals/
├── efficiency_initiative/              # From original tool
│   └── proposal_*.json
├── hicep_dap/                          # From HiCEP-DAP document ✨
│   └── proposal_from_hierarchical_*.json
└── generated/                          # Default output directory
    └── *.json
```

---

## Understanding the Output

Each generated proposal contains:

```json
{
  "source_document": {
    "title": "Document title",
    "path": "Path to source",
    "size": "File statistics"
  },
  "generated_date": "2025-12-10T...",
  "agent_responses": {
    "python_engineer": "Full analysis...",
    "ai_algorithms": "Full analysis...",
    "systems_design": "Full analysis...",
    "training_pipeline": "Full analysis...",
    "architecture_design": "Full analysis..."
  },
  "synthesis": {
    "title": "Research title",
    "core_innovation": "Summary...",
    "implementation": "Approach...",
    "architecture": "Design...",
    "training_pipeline": "Strategy...",
    "system_design": "Infrastructure..."
  }
}
```

---

## Use Cases

### 1. **Refine Your Research Ideas**
You have a rough concept → Generate structured proposal → Get multi-perspective analysis

```bash
python generate_proposals_enhanced.py \
  --mode document \
  --doc my_research_idea.md \
  --output-dir research/proposals/my_project
```

### 2. **Explore Multiple Directions**
Generate proposals for several ideas at once

```bash
python generate_proposals_enhanced.py \
  --mode document \
  --doc idea1.md \
  --doc idea2.md \
  --doc idea3.md
```

### 3. **Automated Research Planning**
Use predefined topics for systematic exploration

```bash
python generate_proposals_enhanced.py --mode auto
```

### 4. **Comprehensive Research Sprint**
Combine your ideas with predefined topics

```bash
python generate_proposals_enhanced.py \
  --mode both \
  --doc hicep_dap.md
```

---

## Tips for Best Results

### 📝 Document Formatting

Your markdown documents should include:
- Clear title (first `#` heading)
- Core innovation section
- Technical details
- Expected gains/outcomes
- Risks and limitations

**Example Structure:**
```markdown
# Your Research Title

## Core Innovation
Describe your key idea...

## Technical Approach
Explain how it works...

## Expected Benefits
- Benefit 1
- Benefit 2

## Challenges
- Challenge 1
- Challenge 2
```

### ⚡ Performance

- Each agent takes ~3-5 minutes
- 5 agents run in parallel
- Total time: ~5-7 minutes per document
- Multiple documents: processed sequentially

### 💾 Output Management

Specify custom output directories:
```bash
--output-dir research/proposals/project_name
```

---

## Integration with AQL v2.0

The enhanced generator works **perfectly** with our existing infrastructure:

```bash
# Step 1: Generate proposal from your research idea
python generate_proposals_enhanced.py \
  --mode document \
  --doc your_idea.md \
  --output-dir research/proposals/your_project

# Step 2: Review the multi-agent analysis
cat research/proposals/your_project/*.json | jq '.synthesis'

# Step 3: Implement using our existing framework
# - Use AQL v2.0 components (experiments/aql_v2/)
# - Test on WikiText-103 (data/wikitext/)
# - Compare with baselines

# Step 4: Document results
# Add to research/sessions/
```

---

## Next Steps

### For HiCEP-DAP Specifically:

1. **Review the Generated Proposal**
   ```bash
   cd research/proposals/hicep_dap
   cat *.json | jq '.' | less
   ```

2. **Extract Key Implementation Details**
   - Check each agent's specific recommendations
   - Identify common themes across agents
   - Note feasibility concerns

3. **Compare with AQL v2.0**
   - HiCEP-DAP focuses on attention pruning
   - AQL v2.0 focuses on data selection
   - **Potential synergy**: Combine both approaches!

4. **Prototype Implementation**
   - Start with Hierarchical Chunk Encoder
   - Add to `experiments/` directory
   - Test on small scale first

---

## Command Reference

```bash
# Help
python generate_proposals_enhanced.py --help

# Document mode (single)
python generate_proposals_enhanced.py --mode document --doc FILE.md

# Document mode (multiple)
python generate_proposals_enhanced.py --mode document --doc FILE1.md --doc FILE2.md

# Auto mode (predefined topics)
python generate_proposals_enhanced.py --mode auto

# Both modes
python generate_proposals_enhanced.py --mode both --doc FILE.md

# Custom output directory
python generate_proposals_enhanced.py --mode document --doc FILE.md --output-dir DIR
```

---

## Comparison: Original vs Enhanced

| Feature | Original | Enhanced |
|---------|----------|----------|
| Predefined topics | ✅ | ✅ |
| Document input | ❌ | ✅ |
| Multiple documents | ❌ | ✅ |
| Custom output dir | ❌ | ✅ |
| CLI interface | Basic | Advanced |
| Flexible modes | ❌ | ✅ |

**Recommendation:** Use **enhanced tool** for maximum flexibility!

---

## FAQ

**Q: Can I use both tools?**  
A: Yes! They're compatible. Original is simpler for predefined topics.

**Q: What document formats are supported?**  
A: Markdown (`.md`) files. Plain text works but formatting helps.

**Q: How long does it take?**  
A: ~5-7 minutes per document (5 agents in parallel).

**Q: Can I customize the agents' analysis?**  
A: Edit the query in `generate_from_document()` function.

**Q: Does it work with very long documents?**  
A: Yes, but first 5000 characters are used to avoid token limits.

**Q: Can I integrate this with my own workflow?**  
A: Absolutely! The JSON output is structured for easy parsing.

---

**Status:** ✅ **Both Tools Ready!**

Use whichever fits your workflow:
- **Quick exploration** → Original tool
- **Custom research** → Enhanced tool
- **Best of both** → Enhanced tool in `both` mode

🚀 **Ready to transform your research ideas into structured proposals!**
