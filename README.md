# benzaiten

## Description

These codes are for 2nd benzaiten contest.

## Model

[PDF](/contents/benzaiten_2nd_yskn67_lt.pdf)

## Dataset

- [Lakh MIDI Dataset(LMD-matched)](https://colinraffel.com/projects/lmd/)
  - This dataset is distributed with a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  - Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
  - This dataset is distributed with a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  - Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." In International Conference on Learning Representations, 2019.
- [Infinite Bach](https://github.com/jamesrobertlloyd/infinite-bach)
- [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html)
  - This dataset is distributed with a the Open Data Commons [Open DataBase License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
  - Pfleiderer, Martin and Frieler, Klaus and Abeser, Jakob and Zaddach, Wolf-Georg and Burkhart, Benjamin. "Inside the Jazzomat - New Perspectives for Jazz Research" Schott Campus, 2017.
- [Charlie Parker's Omnibook MusicXML data](https://homepages.loria.fr/evincent/omnibook/)
  - This dataset is distributed with a [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/).
  - Ken Deguernel, Emmanuel Vincent, and Gerard Assayag. "Using Multidimensional Sequences for Improvisation in the OMax Paradigm", in Proceedings of the 13th Sound and Music Computing Conference, 2016.
- [OpenEWLD](https://github.com/00sapo/OpenEWLD)
  - The compressed MusicXML dataset is distributed with a [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

## How to run

### training

```bash
bash bin/train.sh
```

### generate

```bash
bash bin/generate.sh
```

### generate in contest

```bash
bash bin/generate_multiple.sh
```

## License

These codes are licensed under MIT License without specific copyright holder.

[MIT License](/LICENSE)

Some codes are licensed under MIT License.

Copyright (C) 2023 北原 鉄朗 (Tetsuro Kitahara)

[MIT License](/LICENSE.kitahara)