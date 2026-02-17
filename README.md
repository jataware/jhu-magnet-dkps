This repository contains an example TA1 algorithm integrated with the
MAGNET evaluation framework.

The example algorithm, provided by the JHU team, predicts whether or
not a model will produce the correct answer for a given question based
on the performance of similar models in Data Kernel Perspective Space (DKPS) [1]

## Setup

### Python environment

We use the `uv` tool for environment management.  Installation instructions:
https://docs.astral.sh/uv/#installation

```
uv venv --python 3.11 --seed .venv-311-example
source .venv-311-example/bin/activate
uv pip install .
```

### Downloading HELM results

The example evaluation card requires precomputed HELM results on the
helm-lite benchmark (specifically for the `med_qa` scenario).  If you
do not already have these downloaded, the magnet framework provides a
download utility for these results.  You can run the follow commands
to download and link them into a single `_all` directory.

```
mkdir -p data/crfm-helm-public/lite/benchmark_output/runs/_all

magnet download helm --benchmark=lite --list-versions | while read version; do
    magnet download helm data/crfm-helm-public --benchmark=lite --version="$version" --runs "regex:med_qa.*"
    (cd data/crfm-helm-public/lite/benchmark_output/runs/_all && ln -s "../$version"/* .)
done
```

If you do already have them downloaded but not linked into a single
`_all` directory you can run the following:

```
mkdir -p data/crfm-helm-public/lite/benchmark_output/runs/_all
cd data/crfm-helm-public/lite/benchmark_output/runs/_all
ln -s /path/to/existing/helm/lite/runs/*/med_qa* .
cd -
```

(Note: as there are some duplicate runs across versions, it's safe to
ignore warnings about files / links already existing when running the
`ln` command)

## Running the card

Now that we have the package installed, we can run `magnet evaluate`
on the example card..

```
magnet evaluate jhu_ta1/cards/jhu_instance_predict_auc.yaml
```

In the log output from the process, you should indications of symbols
from the evaluation card being resolved, e.g.:

```
...
Resolving: predictions
Resolving: prediction_comparisons
                        run_spec instance_id  prediction_id                 stat_name  predicted_mean  actual_mean
0   med_qa:model=allenai_olmo-7b     id11718              0               exact_match           0.863          1.0
1   med_qa:model=allenai_olmo-7b     id11718              0         quasi_exact_match           0.863          1.0
2   med_qa:model=allenai_olmo-7b     id11718              0        prefix_exact_match           0.863          1.0
3   med_qa:model=allenai_olmo-7b     id11718              0  quasi_prefix_exact_match           0.863          1.0
4   med_qa:model=allenai_olmo-7b     id11638              1               exact_match           0.642          1.0
5   med_qa:model=allenai_olmo-7b     id11638              1         quasi_exact_match           0.642          1.0
6   med_qa:model=allenai_olmo-7b     id11638              1        prefix_exact_match           0.642          1.0
7   med_qa:model=allenai_olmo-7b     id11638              1  quasi_prefix_exact_match           0.642          1.0
8   med_qa:model=allenai_olmo-7b     id10848              2               exact_match           0.909          0.0
9   med_qa:model=allenai_olmo-7b     id10848              2         quasi_exact_match           0.909          0.0
10  med_qa:model=allenai_olmo-7b     id10848              2        prefix_exact_match           0.909          0.0
11  med_qa:model=allenai_olmo-7b     id10848              2  quasi_prefix_exact_match           0.909          0.0
12  med_qa:model=allenai_olmo-7b     id12252              3               exact_match           0.488          0.0
13  med_qa:model=allenai_olmo-7b     id12252              3         quasi_exact_match           0.488          0.0
14  med_qa:model=allenai_olmo-7b     id12252              3        prefix_exact_match           0.488          0.0
15  med_qa:model=allenai_olmo-7b     id12252              3  quasi_prefix_exact_match           0.488          0.0
16  med_qa:model=allenai_olmo-7b     id12245              4               exact_match           0.467          0.0
17  med_qa:model=allenai_olmo-7b     id12245              4         quasi_exact_match           0.467          0.0
18  med_qa:model=allenai_olmo-7b     id12245              4        prefix_exact_match           0.467          0.0
19  med_qa:model=allenai_olmo-7b     id12245              4  quasi_prefix_exact_match           0.467          0.0
20  med_qa:model=allenai_olmo-7b     id11697              5               exact_match           0.788          1.0
21  med_qa:model=allenai_olmo-7b     id11697              5         quasi_exact_match           0.788          1.0
22  med_qa:model=allenai_olmo-7b     id11697              5        prefix_exact_match           0.788          1.0
23  med_qa:model=allenai_olmo-7b     id11697              5  quasi_prefix_exact_match           0.788          1.0
24  med_qa:model=allenai_olmo-7b     id11891              6               exact_match           0.802          0.0
25  med_qa:model=allenai_olmo-7b     id11891              6         quasi_exact_match           0.802          0.0
26  med_qa:model=allenai_olmo-7b     id11891              6        prefix_exact_match           0.802          0.0
27  med_qa:model=allenai_olmo-7b     id11891              6  quasi_prefix_exact_match           0.802          0.0
28  med_qa:model=allenai_olmo-7b     id12054              7               exact_match           0.524          0.0
29  med_qa:model=allenai_olmo-7b     id12054              7         quasi_exact_match           0.524          0.0
30  med_qa:model=allenai_olmo-7b     id12054              7        prefix_exact_match           0.524          0.0
31  med_qa:model=allenai_olmo-7b     id12054              7  quasi_prefix_exact_match           0.524          0.0
Resolving: compute_auc
Resolving: computed_auc
...
```

(Note: it's safe to ignore warnings about "dkps.embed: unable to load google-genai")

Once the card has been fully evaluated, you should see the following:

```
================================
Settings Evaluated: 3
  Verified:     1.00
  Falsified:    0.00
  Inconclusive: 0.00
================================


Title:       JHU DKPS based per-instance metric prediction
Description: We can predict whether a particular model will produce the correct output based on the performance of similar models in Data Kernel Perspective Space (DKPS)

================================
CLAIM:       
assert computed_auc > auc_threshold, assert_failed_msg

================================
RESULT:      VERIFIED
================================
CARD STATUS: EVALUATED
```

This output indicates that three variations of the evaluation card
have been evaluated (the example card sweeps over three different seed
values for random evaluation set selection).  In this case all three
variations have been verified (claim passed), so the final `RESULT` of
the card is that it is `"VERIFIED"`.

## Citations

[1] Hayden Helm, Aranyak Acharyya, Youngser Park, Brandon Duderstadt, and Carey Priebe. 2025. Statistical inference on black-box generative models in the data kernel perspective space. In Findings of the Association for Computational Linguistics: ACL 2025, pages 3955–3970, Vienna, Austria. Association for Computational Linguistics.
