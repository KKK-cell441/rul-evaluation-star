# Zenodo v18 Upload Checklist

Use this checklist when creating the v18 Zenodo release. After publishing, send the new DOI back to Codex so the manuscript, cover letters, and repository docs can be updated.

## Upload destination

- URL: https://zenodo.org/upload
- Account action required: publish the upload and obtain the new DOI.

## Metadata

- Title: Reconsidering Reliability Assessment of Data-Driven Predictive Maintenance: The Impact of Evaluation Protocols on Bearing Remaining Useful Life Prediction
- Authors: Zhongkuan Ma
- ORCID: 0009-0003-4491-6619
- Version: v18
- License: Creative Commons Attribution 4.0 International (recommended)
- Description: Reproducibility package and submission candidate for the bearing RUL evaluation protocol study.

## Files to upload

- `manuscript_revised_v18_jim.pdf`
- `manuscript_revised_v18_jim.tex`
- `appendix_perfold.tex`
- `README.md`
- `Submission_Readiness_Checklist.md`
- `Zenodo_Upload_Checklist.md`
- `figures/graphical_abstract.png`
- `figures/graphical_abstract.pdf`
- `reproducibility_v18.zip`

## After publishing

1. Copy the new DOI, e.g. `10.5281/zenodo.XXXXXXX`.
2. Send the DOI to Codex.
3. Codex will:
   - restore the DOI in `manuscript_revised_v18_jim.tex`;
   - restore the DOI in `Cover_Letter_JIM.md` and `Cover_Letter_Measurement.md`;
   - update `Submission_Readiness_Checklist.md` and `reproducibility/README.md`;
   - recompile the PDF, sync `0809`, and push to GitHub `main`;
   - verify the DOI resolves publicly.
