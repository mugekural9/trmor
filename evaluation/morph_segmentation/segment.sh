#! /usr/bin/env bash

sample_types="subword_given word_given"
recon_types="avg sum"
model_id="VAE_FINAL"

echo $model_id
for sample_type in $sample_types; do
	echo $sample_type
    for recon_type in $recon_types; do
        echo $recon_type
        python -W ignore /kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/vae_segment.py \
        --recon_type $recon_type \
        --sample_type $sample_type \
        --model_id $model_id
        perl evaluation/morph_segmentation/evaluation.perl -desired evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur  -suggested /kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/results/vae/$model_id/$recon_type/nsamples10000/$sample_type/prev_mid_next/eps0.0/segments.txt
    done
done
