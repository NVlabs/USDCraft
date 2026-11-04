# Upstream license texts

License texts for dependencies in which `tools/third_party_notices.py` found no suitable license file
in the locations it inspects (a Python wheel's `.dist-info` directory, an npm package's root). This
does not mean the package as a whole carries no license text: some wheels keep one elsewhere in
the package. Each text here was taken from the upstream source repository (at the matching release
tag where one exists) and is copied by the generator next to the package's entry. Sources:

| Package | Source |
| --- | --- |
| cadquery-ocp 7.9.3.1.1 | <https://github.com/CadQuery/OCP/blob/7.9.3.1/LICENSE> |
| cadquery-ocp-proxy 7.9.3.1.1 | Standard Apache License 2.0 text (the package declares `License: Apache-2.0`; author Bernhard Walter) |
| casadi 3.7.2 | <https://github.com/casadi/casadi/blob/3.7.2/LICENSE.txt> |
| colorama 0.4.6 | <https://github.com/tartley/colorama/blob/0.4.6/LICENSE.txt> |
| cython 3.2.8 | <https://github.com/cython/cython/blob/3.2.8/LICENSE.txt> |
| format 0.2.2 | Standard MIT text with the copyright line and license named in the package `Readme.md` |
| highlightjs-vue 1.0.0 | <https://github.com/highlightjs/highlightjs-vue/blob/master/LICENSE> (BSD-3-Clause; the npm package declares CC0-1.0) |
| react-remove-scroll-bar 2.3.8 | <https://github.com/theKashey/react-remove-scroll-bar/blob/master/LICENSE> |
