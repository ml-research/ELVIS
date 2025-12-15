# Quick Reference: Using Heatmap Captions in Your Paper

## 📋 Copy-Paste Ready Captions

### For Individual Principle Heatmaps

```
Figure X: Raw Factor-Level Performance for [Principle Name]. 
Heatmap showing mean F1 scores across experimental factors for different 
task categories. Rows represent task categories; columns represent factors 
(relevance, group counts, sizes, rules). Empty cells indicate factors not 
manipulated in that category. The "Mean" row shows average across categories; 
the "Mean" column shows average across factors. Color scale: 0 (low) to 1 (high).
```

### For Merged All-Principles Heatmap

```
Figure X: Factor-Level Performance Across All Gestalt Principles. 
Comprehensive heatmap showing mean F1 scores for experimental factors across 
all five Gestalt principles. Rows represent task categories grouped by 
principle (separated by black dashed lines). Empty cells indicate factors 
not manipulated in that category (by design). "Overall_Mean" row shows 
performance aggregated across all principles. Color scale: 0 (purple/low)
to 1 (yellow/high).
```

## 📝 In-Text Description Template

```
Figure X presents [describe what you see, e.g., "factor-level performance 
analysis"]. The sparse structure, with empty cells indicating factors not 
manipulated in each category, reflects our principle-specific experimental 
design. Notably, [mention key finding, e.g., "performance on rel_shape 
factors is consistently lower than rel_color across all principles"], 
suggesting [your interpretation].
```

## 🔬 Methods Section Text

```
We analyzed factor-level performance by computing mean F1 scores for each 
experimental factor (shape/color/size relevance, group counts 1-4, object 
sizes s/m/l/xl, rule types) across task categories. Not all factors were 
manipulated in every category; only theoretically meaningful combinations 
were tested. Performance metrics were aggregated both across factors (row-wise 
means) and across categories (column-wise means), with all means computed 
only over non-empty cells.
```

## 🎯 Key Points to Remember

1. **Empty cells = Intentional design** (not missing data)
2. **Emphasize**: "Only theoretically meaningful factor combinations tested"
3. **Means computed**: Only over non-empty cells
4. **Visual separators**: Black dashed lines separate principles

## 📊 File Locations

Individual: `baseline_results/figures/task_factor_analysis_{principle}.pdf`
Merged: `baseline_results/figures/task_factor_analysis_all_principles_merged_gpt5.pdf`

## ✅ Checklist for Paper Submission

- [ ] Include figure in paper
- [ ] Add caption from above
- [ ] Mention in main text
- [ ] Explain in methods section
- [ ] Highlight key findings
- [ ] Note: Empty cells are by design

---

**Everything is ready! Your heatmaps now have professional captions explaining the sparse structure.** 🎉

