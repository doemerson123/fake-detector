# Congratulations!!
## You've just spotted your first deepfake
Okay probably not - they're everywhere. That's what this project is about. 
1. Using technology to spot deepfakes 
2. Learning how to spot them more easily without the aid of technology using explainable AI



![alt text](https://i.insider.com/5c6d85ca2628986f7f3a5d02?width=1000)


## TLDR: 
This project trains lots of different convolutional neural networks (CNNs) and tracks the experiments to find the best model to spot deepfakes - from scratch (without MLFlow or gridsearch). The best model is ![shared online](https://donovinemerson.com/?p=273) (manually for now). As an added bonus, this model will train *YOU* how **not** to not need it by explaining what it finds most important about the image for its prediction. 

There were four convolution layers in the best model. We see each one light up green where the model found the most important information (eyes, structural area, outlines, and teeth). Protip: the areas with the highest geometric variablility are where to look first.

![image](https://user-images.githubusercontent.com/87036676/216816048-caa696c7-1128-4d86-8a95-82f0bc1d2a01.png)

PS: Apologies to any visitors who are red/green colorblind! Unfortunately this color scheme was so much better than the other choices. 


## Diving Deeper:
That's not all that's been done in this project but we'll get into that, well... now.

Model training was conducted on ![StyleGAN generated images](https://arxiv.org/pdf/1812.04948.pdf) from http://this-person-does-not-exist.com. The benchmark "real" human pictures are from the ![Flicker Faces dataset FFHQ](https://github.com/NVlabs/ffhq-dataset). 
Performance is excellent for this dataset (F1 score >99%) but the zero shot cases from other known deepfake images do not fare so well. I'm working to make the 55GB dataset available but have not successed - yet!

## Repo Contents
Contained in this project are two applications 
1. Webscraper/webcrawler that gathers deefakes to train models
2. A custom OS independint experiment tracker
    - Ingests a configuration file to manage artifacts, the data pipeline, and model parameters - **params.yaml**
    - Permutes model parameters (cartesian product) to create the list of experiments
    - Trains and evaluates each of the models
        -  For each model, a new folder is created in the artifacts directory to store plots and saved model files (pb, hdf5, h5)
        -  After all models are trained, an aggregate DataFrame with details of each experiment (parameters and results) is stored in the main artifacts directory
    

## NOT Contained in this Repo
This model is available to interact with on my ![personal website](https://donovinemerson.com/?p=273) and is served using Streamlit - please play with it and let me know what you think! Link to that repo is ![here](https://github.com/doemerson123/fake-detector-api)


## Project Tree:

Most logic resides in the utils files. The deepfake scraper and testing folders are stand alone however model_training.py in the root directory relies on all the files in the utils folder. 

    |- deepfake_scraper
        |- data_collection.py ****Entrypoint for webscraping
        |- webscraping_util.py
    |- test
        |- Fake
            |- test_Fake 4.jpg
        |- conftest.py
        |- test_data_pipeline.py
        |- test_modeling_utils.py
        |- test_params.yaml
    |- utils
        |- custom_metrics_utils.py
        |- data_pipeline_utils.py
        |- modeling utils.py
        |- plot_metrics_utils.py
    |- model_training.py ****Entrypoint for model training
    |- params.yaml
    |- Dockerfile
    |- Makefile
    |- environment.yml
    |- requirements.txt
    |- README.md


## Is this thing on????
Since all configuration for the app occurs in params.yaml, entry points do not have a CLI. Executing either of the two .py files/modules in the terminal will trigger the logic: `python model_training.py` or `python data_collection.py` 

Test strategy is implemented using pytest. All code and supporting files reside in the test folder. Calling pytest in your favorite way will bring it to life.



how to kickoff training

high level overview of scraper and how it works

links to docs directory technical details for all utils


```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```
```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```
```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```
```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```
```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```
```
fake-detector
├─ .dockerignore
├─ .git
│  ├─ branches
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     ├─ remotes
│  │     │  └─ origin
│  │     │     ├─ main
│  │     │     └─ master
│  │     └─ stash
│  ├─ objects
│  │  ├─ 01
│  │  │  ├─ 55da574d9d76046b17efd8ed1b5995fd9a35c7
│  │  │  └─ f8f07ae86f623240ccc823cce9abec91b78193
│  │  ├─ 02
│  │  │  └─ a59520bd9544301f5ec7e969e1bbd940d7a819
│  │  ├─ 03
│  │  │  └─ fcc7126b69ca3541ce5f95245f96516c01d701
│  │  ├─ 05
│  │  │  └─ 8a64a5736191dcf232209c4dee0d7cfe089ace
│  │  ├─ 06
│  │  │  ├─ 12b773c405875acc593ec704fb1b64c0da6134
│  │  │  ├─ 1c57f065c31ec4478645fce71c3dd7dd2efda0
│  │  │  ├─ 5703160206db4f92d4743be2753a5173ccce32
│  │  │  └─ b6897bb1012bb5deb3cfceae7ce76ab63b653d
│  │  ├─ 07
│  │  │  └─ 3f87dc19246f25e5154fb81b0b7ae187b95dc3
│  │  ├─ 08
│  │  │  ├─ 1662e65b82cbdc2a14fd8634158e0c3de8dc54
│  │  │  └─ 2355da855143fd092b6a0ccf21f37b840362a1
│  │  ├─ 0b
│  │  │  ├─ 48070c67c64fd32fe614475723df8317d1d113
│  │  │  └─ e4b06289a489902818e783b28d9ae7e602657f
│  │  ├─ 0c
│  │  │  ├─ 07fdd59f3e173987848afd9946367b1a170a0b
│  │  │  └─ 9219600bd3a6b6174ca46c41fe3b3f10792228
│  │  ├─ 0d
│  │  │  └─ 248956f2369d9c9065bc3b8472d9cd4bd69a0c
│  │  ├─ 10
│  │  │  └─ 5881f0befc79192a57f436cc9a9fb48cca5ca4
│  │  ├─ 12
│  │  │  └─ 70a11e98735b597ae0274e886acb8a0bed4369
│  │  ├─ 13
│  │  │  └─ cfa811145f8c0e136d6c7459e059fc88aa82d6
│  │  ├─ 14
│  │  │  └─ fb6b0409b7e6cc04e9e69ea2b2543ebe9916fd
│  │  ├─ 15
│  │  │  ├─ 0e43e15b7b24ef20f6f307b51c3a74bb99615e
│  │  │  ├─ 2bce74bea6e90272ece3d6cc389193221df85d
│  │  │  └─ 7740853a4bb7c0d78c2835716a0490191908d1
│  │  ├─ 17
│  │  │  └─ d1fa76ba0fdab0d92b744a9dea01bf03b92e63
│  │  ├─ 19
│  │  │  └─ c933d7a77d32b9deb7f15c4eafd67e3cb294d8
│  │  ├─ 1a
│  │  │  └─ 598072ecd0471c240d0b4da838658d5d7f8aa8
│  │  ├─ 1b
│  │  │  ├─ 6080d05ab43c8b04bd3d73ea81f5c4e8c0e493
│  │  │  └─ df6e9e030f9016999945f2690c5f0fd4b90420
│  │  ├─ 1e
│  │  │  └─ 9714200dc39cc5d0ce15eaf3e1836d0a06c77d
│  │  ├─ 1f
│  │  │  └─ 7ac75c76b8fe8a889ffe6232eb09e5efc7500c
│  │  ├─ 20
│  │  │  ├─ 65ac02f605f581f099c985c27d00fc6c0bed30
│  │  │  ├─ 77fd889c475dda2273782ebb4fa4d8878807f8
│  │  │  └─ e3dd491c8e114212a3f0d8f27c425e6019c567
│  │  ├─ 21
│  │  │  └─ 21b2891d9105069d94c24acc13d148f11fe785
│  │  ├─ 24
│  │  │  └─ ec778c4f7b1683f692f54cb816cace8f617c1a
│  │  ├─ 29
│  │  │  └─ 7ad01030ab1bfc8a92f88d5d5366caeda7d97d
│  │  ├─ 2b
│  │  │  └─ fd14ac29236cfe8f87f313cbf329d2308b1a69
│  │  ├─ 2c
│  │  │  ├─ 5be1c8ce7797b83d2a7842d65f85bf5915be13
│  │  │  ├─ 7974f2565575f90e264067d9e78484cc040aa5
│  │  │  └─ 93c5025aaff274e5adf00564c6c4fb28f1655b
│  │  ├─ 2f
│  │  │  └─ 5235b29d10a2129233557aaeb0be9ccbeffdec
│  │  ├─ 31
│  │  │  └─ 95ce1996588ddba9c5ae17301ee340c68f0fc5
│  │  ├─ 32
│  │  │  └─ 6a3ffbb6afa2f3fb36afee3212a30082e8b324
│  │  ├─ 34
│  │  │  └─ 7b81628827dd8f2969d7969a8c948d11339a1a
│  │  ├─ 35
│  │  │  └─ 797d4b6ad958ec4c295a33bd7c878d0ccf310d
│  │  ├─ 3a
│  │  │  └─ 6f1d1cd841f67040413aa08e538cee1c7bf609
│  │  ├─ 3c
│  │  │  └─ 4a2bf191af187023ab8286cb4a0a28d85404ca
│  │  ├─ 3d
│  │  │  └─ 29197217a0f609a9a334ea244bc177d1b91adf
│  │  ├─ 3e
│  │  │  └─ c7bddc22038197a01c11d71ddd0cd235a456f3
│  │  ├─ 3f
│  │  │  └─ fc56f5d6f1beb506518f7d5f8cc5d15dd6645c
│  │  ├─ 41
│  │  │  └─ e8a24073b9c9e70e0bb3b9ef17d49d5cc767ad
│  │  ├─ 43
│  │  │  ├─ 7a38a28d61f7ddd9e6e6bd17dadaf3ac8b4494
│  │  │  └─ ac9ff8995d64cf2f774bfa2c0cf95e3172bd30
│  │  ├─ 45
│  │  │  ├─ 7f940afb47625a40016bea055a518b3c77ca44
│  │  │  └─ 85d23231c15299a4e66011be429356513e4ddf
│  │  ├─ 46
│  │  │  ├─ 5fa0ba83a028da1d6e00c4a40e35cb57919bbd
│  │  │  └─ febbbcc402c4e4a91632e1ec32e1e28e3f92fe
│  │  ├─ 49
│  │  │  └─ 7218106b7152cea748e744ab7989402ec599a8
│  │  ├─ 4a
│  │  │  ├─ 0bfe306e7ed67c19c3c8995c5739ab78d86644
│  │  │  ├─ 2685f1d4f00ccfe9387a65620207c217e5334a
│  │  │  ├─ 6504702b0f2d64883a8b2cfa0a2b3547d020b1
│  │  │  ├─ 906d890f4ad9a01299eb69e82139a003010e09
│  │  │  └─ d63fe8219bcf189f3423f1def41e99181a65c9
│  │  ├─ 4b
│  │  │  └─ 8a8b5acef9e5d86d4e8e9740b0f77d5afdd8f9
│  │  ├─ 4c
│  │  │  ├─ 4ba3559585296934e4269cf6ed041d0031ce92
│  │  │  ├─ 7f87cac97e1bd2f55e6b16d74752ab940d09d6
│  │  │  └─ bd45ba176b175819b8f3920e29252f939e5af8
│  │  ├─ 4d
│  │  │  └─ 4fbc2b41abb3abf568258b1c332c9e224607ef
│  │  ├─ 4e
│  │  │  └─ 4b3c926a394cd91c426366fbeca99b6eec6dd8
│  │  ├─ 51
│  │  │  └─ f9e8dd0e1c2473d413b4f7dc8ee37ed05c7b60
│  │  ├─ 54
│  │  │  └─ 0e002d5845a3256d5324d2418fbf1d47d2857c
│  │  ├─ 55
│  │  │  ├─ 1cea09f7e3b38f8ae60175dfaa41fd2b005108
│  │  │  ├─ 3f7e877bee2a8415793bb4f8a8cdcbdfe36e4a
│  │  │  ├─ a8a12dadcdf63f49ed70838c2ecf5a1bc1a405
│  │  │  └─ f802e962e9d77f0035be29c167dea214608916
│  │  ├─ 57
│  │  │  └─ 070d591a5dc193099d1afa62f0cd6acbfc1952
│  │  ├─ 58
│  │  │  └─ fb4d0e5b4f7957b35e16ab703b4c89227e1652
│  │  ├─ 5a
│  │  │  └─ 8d3f5f01f3160b59e3545adcdb832188792c87
│  │  ├─ 5b
│  │  │  └─ ed5dd15209c01f8fb3d58c1764c07439bcf5be
│  │  ├─ 5c
│  │  │  ├─ 0beb7cfbadbf89e14e3306cf2636685910e216
│  │  │  └─ 9ce3e4d2d0621fea340179d9f28fb93cac7405
│  │  ├─ 5d
│  │  │  ├─ 3ea1da04ff3702b593be4ef71416e7b4767129
│  │  │  ├─ 9cb91c43341c24f00ae7378645d33fcd7c61b6
│  │  │  └─ d438c43a65b647220d6fd3b52558a80bab770f
│  │  ├─ 60
│  │  │  └─ 3a664c3b26c9ce3e6862056509c9c8031c1c9d
│  │  ├─ 61
│  │  │  └─ 64c2f27e7d7e3831a0d82a39bcb4ca1e6fac29
│  │  ├─ 62
│  │  │  └─ 16728762c9389305b3f44f8887787dcc30bbe9
│  │  ├─ 63
│  │  │  └─ 065c8899f68e8e8f3d6e70d1a23d83d5897d5e
│  │  ├─ 64
│  │  │  ├─ 2ebd0f5c341a3178d313afb8b9f29c5eb66383
│  │  │  └─ 8bc917b1b828f157e3e7210fec7bfda212f448
│  │  ├─ 65
│  │  │  └─ e0ae24e6ff8a3e2c373a43fe92246164228603
│  │  ├─ 67
│  │  │  ├─ 52cfeafd68dc7ecd9ffb64f65c43f23c0535a6
│  │  │  └─ 8a010f5c9c6f110fe7b562c23ed9de13e754bc
│  │  ├─ 69
│  │  │  ├─ 271c75d156212ff7c1dc1396b9394adac801c7
│  │  │  ├─ 3ddbf96d01356dcd94fc8efa70b2a0b5a32a5d
│  │  │  └─ e0d94861354cdc431995613aa4edf931ff9335
│  │  ├─ 6d
│  │  │  └─ cfc92c542005b767c0a0ded8d355b7a918a993
│  │  ├─ 70
│  │  │  ├─ 1a3654b5654d64532a002fb6db2e2ff6829168
│  │  │  └─ 714f8c579d8266dfda6e2e9a9a4caaf6bfbb4b
│  │  ├─ 72
│  │  │  ├─ 167d72f929efb015d3d46ed6ec37f89bd29802
│  │  │  └─ 5bac50239452c4b34ffaded47042b223a6f008
│  │  ├─ 74
│  │  │  └─ b6dceb23a89676fa073ea7407560508a468fdc
│  │  ├─ 75
│  │  │  └─ 04dfbe27dc2cbf485a50631fb988f9b87c66c0
│  │  ├─ 76
│  │  │  ├─ a437b665a9fcd4f6340272dbb7f48583e454c9
│  │  │  └─ df3f9c24f924fa0673c263f26dfb3562a350d4
│  │  ├─ 79
│  │  │  ├─ dbd8909dbaf9091761095025e4fa55a490900e
│  │  │  └─ f2def72744d4e629891c37e07a9f31a8208e77
│  │  ├─ 7a
│  │  │  ├─ 82b10fd02848f90768f508988d35f561ec174c
│  │  │  └─ ed34b9920a0498a11a5eed5fbbb1b4f7c0875a
│  │  ├─ 7b
│  │  │  └─ 29fef8777e99ac5f3aabd3bd3ee982a3ec46ed
│  │  ├─ 81
│  │  │  └─ 3d276a3b7c8b376901b3b412bd90e80edc256f
│  │  ├─ 82
│  │  │  ├─ 19e116947ef0731c945d3b432270a2422a382a
│  │  │  └─ ed2f609e9ea39e411ccb7ae2aa83299b5aff4a
│  │  ├─ 83
│  │  │  ├─ 6c24c1b62172a86603819994d7c0f69fbe6f9a
│  │  │  └─ d992d9d2a60341e51bbf2c59e3e480ea0ffc5f
│  │  ├─ 84
│  │  │  └─ 55e73c40e9df737ee4ea6d6dab7b8a538937bd
│  │  ├─ 87
│  │  │  └─ 343142407f029cc5979cb3fb1f2f95484fd741
│  │  ├─ 89
│  │  │  ├─ a604a497a71c6abed5c4a85cb375794fcb3be7
│  │  │  └─ dbd18f59b679abf0b5d229a942d5bb1b8333b1
│  │  ├─ 8a
│  │  │  └─ 2494522eededdc1def7946b7cbd077fd635bcf
│  │  ├─ 8d
│  │  │  └─ 980a9ae829afdf99d4bcd64f481131a52b150e
│  │  ├─ 8e
│  │  │  └─ ec84cfbfcb537e9936b3fe08dc6aff84c578dd
│  │  ├─ 90
│  │  │  └─ b5f05d7fc399fa819396c0d35992546c7b5a3c
│  │  ├─ 94
│  │  │  ├─ 4107b1fc7eadba212e7a715e23d8162d481cfc
│  │  │  └─ e314211eb7d0de71e20da8ac0adec6c686fd60
│  │  ├─ 95
│  │  │  └─ a87473caae3de52f5f0b5c76e48d024b21f080
│  │  ├─ 96
│  │  │  ├─ 45c2c5502c10d465437bf7bd0801412fac3309
│  │  │  ├─ 7f686777dacbc5d0964e958e3d5b7d8987f278
│  │  │  ├─ ce8ad2b31ee340fe18fabf272089c69b72f1ac
│  │  │  └─ e480162cf981ee22d83f6077d4e9b4f10240be
│  │  ├─ 97
│  │  │  └─ b38787ea697a927e2ab483532aed0b79f32eee
│  │  ├─ 98
│  │  │  └─ 57cc853334d2ca8b48e2f79886456331ff170d
│  │  ├─ 99
│  │  │  └─ 2a8b96dd8003925b5074c443a6fc2d94a96c0f
│  │  ├─ 9a
│  │  │  ├─ 2441f51a938da4d9c795a2e73143ab550e8ccd
│  │  │  └─ 4f63370c42479846f4761f6a7b754d271b1723
│  │  ├─ 9b
│  │  │  └─ e63817094d9f1be49905d7344b269834d74dd6
│  │  ├─ 9d
│  │  │  └─ 88e883405535b55d2c8f818bacc47b98978281
│  │  ├─ 9e
│  │  │  └─ 39ee87ab23355c0e5ae1fbe9a0120e5673928b
│  │  ├─ 9f
│  │  │  ├─ 0564a4c9c7d16461eb5b31fba6947e0c5a62b1
│  │  │  ├─ 0c4f827ee6a1dd6ffcf4f9d8c0f44eeb9405ac
│  │  │  └─ c231a8105ad70f89f52ae413ccf2f65d023a3f
│  │  ├─ a1
│  │  │  └─ 2a83cb9aa5b09c87c146f0084270614938d98c
│  │  ├─ a2
│  │  │  └─ f65a2b8d197ae8a6c1a161f36b559862596045
│  │  ├─ a3
│  │  │  ├─ 189827557267bb644a61ba381d82fceb922c9b
│  │  │  └─ b4f50241d11e6cb53552a90c03b9e521ed7be5
│  │  ├─ a4
│  │  │  ├─ 4aa018343f4db42cffc5598b7b03304589eecd
│  │  │  ├─ 9467a062f10efe9e23e7f4567e51abf895598b
│  │  │  └─ fd2143e9d208f23f29e7b174ebf405b72644c8
│  │  ├─ a5
│  │  │  ├─ 12d07a955413f1f417e9a02dfa29ba1edd0aa7
│  │  │  └─ 669c8c44d4be7e1b9a89040677053f77d95383
│  │  ├─ a6
│  │  │  ├─ 257a4c049798c545e7d8d8cb6b656d4cacaaa3
│  │  │  ├─ 8bc37b2ccbae1dd8b0b90a02814370ed6bc951
│  │  │  └─ db91df530c8c1e8bbca8146e5d32e2c95accdb
│  │  ├─ a8
│  │  │  ├─ 44ca5fdb553db6ca414a579cd7cfabedf75f1d
│  │  │  ├─ 77939e6db9f935e373d73d1ba31556d7cb3a08
│  │  │  └─ a66996619e958c42e174075f9c43252c7a61fa
│  │  ├─ a9
│  │  │  └─ 4327cab30112ab3d166b2a28735b94c66d5ecc
│  │  ├─ aa
│  │  │  └─ 2d40f841f25234226a946548080d214d40faa9
│  │  ├─ ac
│  │  │  ├─ 2b2bfc3546b43761f3c19699cede8fa410537e
│  │  │  └─ cedf942873bc7c93e088fadac586750cf59ab8
│  │  ├─ ad
│  │  │  └─ 05b073e2fc3b7a5810a82657f36c937fc2ab25
│  │  ├─ ae
│  │  │  └─ b9ca5f4b543f19562a33d9fd04368b76c126c6
│  │  ├─ b2
│  │  │  └─ e100de87c5ffafadd8bf876b7b0d54da002573
│  │  ├─ b4
│  │  │  └─ 380e372d2dc76653ed68c09876c7c4d6099560
│  │  ├─ b5
│  │  │  ├─ 4ebd3d34dc7580db25b36b71a914baf3f253c9
│  │  │  └─ 5d8f630ead3fd208f1d0c0b0e53835d23a97dc
│  │  ├─ b6
│  │  │  ├─ 0534af50bdad90e093b1a9cbabd908dc62bd18
│  │  │  ├─ 6fee1f492d037f80791de536252f9930be2ea6
│  │  │  └─ b02ea27937f4738fa14218432adc82fef0b4ab
│  │  ├─ b7
│  │  │  └─ d7b06d8dc1c7f564bd6eeb708815f9a3640b4a
│  │  ├─ b8
│  │  │  └─ 08ebdeb3c6c5cc5331ed9b9f05ce9c8b807f02
│  │  ├─ b9
│  │  │  ├─ 727e5b3d45578e04c4fbf573e03cc79e75379c
│  │  │  └─ a71bb9cd0ad4d4bfbbde74ef5b92694f981828
│  │  ├─ bb
│  │  │  ├─ 13a6a8f548d1abd3974c379c763f4c71702738
│  │  │  ├─ 5b4975d97d71d262a7ee17e5af949b0d0bfbff
│  │  │  └─ 9853298541df5730411c950458ad772837befc
│  │  ├─ bc
│  │  │  ├─ 8c71526c1f15a581ac3712c5447f2b5326c1ef
│  │  │  └─ cc5492ad439003906eb5f5daec0c52bd8ec1a1
│  │  ├─ be
│  │  │  └─ ed2c68a43262add69eef1b1c31f2ce72a2b1b9
│  │  ├─ c1
│  │  │  └─ aae3cb7579c603daeb3845ca93656e18f34609
│  │  ├─ c2
│  │  │  └─ 4fa18283073ac4b71aa857cebed6505b28e949
│  │  ├─ c3
│  │  │  └─ 18dc704f4a40b0e65c5aaa88079572ddae0a71
│  │  ├─ c5
│  │  │  └─ 2eff9fe0b7781944b9f922013c0215caf1959c
│  │  ├─ c7
│  │  │  └─ f8620e427afaefa353594ed47451d5e430d2ed
│  │  ├─ c9
│  │  │  └─ f2f0517bb1ce7afdf9db77cc0d28233e453296
│  │  ├─ ca
│  │  │  ├─ ebf93b008326935674cf0e3d9723f83ac87af3
│  │  │  └─ f63291463b614fbd2663a044ba1a1ac4b824c9
│  │  ├─ cb
│  │  │  └─ f9c391799c99f17bb946988d0f41d021f3893a
│  │  ├─ cd
│  │  │  ├─ 42d93d23f64b245194516b17da4386d8fe3d2a
│  │  │  └─ 644c38ac8a5bccec607ae782a6c5c41c1286ab
│  │  ├─ ce
│  │  │  ├─ 0312bcca1ed13d78381a16daa7321c49480d72
│  │  │  └─ a52c13be990dea0549af79b77488ff4779bbf9
│  │  ├─ d0
│  │  │  └─ e4a398727d456648dc7e8073e944665496c5f7
│  │  ├─ d2
│  │  │  └─ 89e5670e0900f1b8fca1f0700ffc51ad4e1a22
│  │  ├─ d4
│  │  │  └─ d0e232aac15e62d835fb334d28d170ae1f5225
│  │  ├─ d5
│  │  │  └─ e12f9c45d4981e28b05c11777f4366f4ace90d
│  │  ├─ d6
│  │  │  └─ 2a375231ca43ef6deb40ed0671023f229ca442
│  │  ├─ d7
│  │  │  └─ 027221f379f7be71b392f739c5074a4d789561
│  │  ├─ d8
│  │  │  └─ 8f5405a8217d0c67914d097118115d64c5e1ed
│  │  ├─ da
│  │  │  ├─ 6f6dd90ab78503be2a0658421dd9c9b887211a
│  │  │  └─ f8a4cd5afe259e0aa952d70ebff79cbfa945ea
│  │  ├─ db
│  │  │  └─ 09295f3439f51ea3d89df49a87181b925f6c5c
│  │  ├─ dd
│  │  │  ├─ 016bb5cf0c30bcc1f62e02dcff87793ea02b65
│  │  │  ├─ 38f49321b585cdd20d841acf3decd1aff95ba6
│  │  │  └─ 7c918380042acd58de35c4bd62cf19c1d67026
│  │  ├─ de
│  │  │  └─ 3b7f588c99f68742cb58ecffb2e73ed7a3a767
│  │  ├─ df
│  │  │  ├─ 00d61daea025adddfe595bb2b44e7751e87ce2
│  │  │  ├─ 7de8f47089f63639271d51ee0a70e905551b5d
│  │  │  └─ 9b7f07257a93563fc6f13306cbca42d89cb77f
│  │  ├─ e0
│  │  │  └─ 5a37a23bc272691a737d6af1e8a76e8053ce93
│  │  ├─ e1
│  │  │  ├─ 13bb4fe250af923bfcf480fd5f9393dbd9fd47
│  │  │  └─ 4d2d810d871f3a9fbc47294ca861f613b3b9cf
│  │  ├─ e2
│  │  │  ├─ 5254cf5d0c1dfeaa01d08703d9d4727aa56baf
│  │  │  └─ a19f59c9729ee973ec14f54bdebe8bc5377d29
│  │  ├─ e3
│  │  │  └─ 4201a6caffd83f973c6eeb2dead80ead8a8518
│  │  ├─ e5
│  │  │  └─ 889d64fea3edc3882d1afd27e91bc0ab80e3bc
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ e8
│  │  │  └─ c92f6fcf1792dee8d99642ab8b2fa1c9c0d1ec
│  │  ├─ e9
│  │  │  └─ 9d672fef68e1af8455aecb3b28465cb7a7c66e
│  │  ├─ ea
│  │  │  ├─ 19356b41937d62f1ae553c25e0af052ffa9be9
│  │  │  └─ 401a5b3a32ba765431723425c091d0eea5e508
│  │  ├─ f1
│  │  │  └─ 936459e3c1c8ef8e8f5ce81417807fb55404f8
│  │  ├─ f3
│  │  │  └─ cdc9368828d8f7c228beec6d691d8904e4280c
│  │  ├─ f4
│  │  │  ├─ 112277797a08e40ec78788aa24fba003f827aa
│  │  │  └─ f9bfe8c410a00d2d82dc627ff7397173b00144
│  │  ├─ f5
│  │  │  └─ bad49b9413a4f61d365f775860d9380e27e5bf
│  │  ├─ f6
│  │  │  └─ 025973a6bc6c24b090c3ad3e81dbead05d1907
│  │  ├─ f8
│  │  │  ├─ 7574d0776ec469912cb89fe1f97c1f55bc1def
│  │  │  └─ ef3b644d708a2a487000bd9add8bd0fbe635be
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fa
│  │  │  └─ 045f0fdfa0cef0ceacbc8b561d61ff3ca8a6ee
│  │  ├─ fb
│  │  │  └─ 2f60943de8297f2d8a51d3b2314dea8ed6ecbf
│  │  ├─ fd
│  │  │  ├─ c2a69732dd8d0fb9808d3e28db9f0ab34a9073
│  │  │  └─ ddfc3735fd4c1c0225b5b6ebc0aed21505ad35
│  │  ├─ fe
│  │  │  ├─ 2538aafaa54b5021eded69c53fa2030f0d69bb
│  │  │  ├─ 5e646133bd1ce36b7a340dab78d366ffd38aae
│  │  │  ├─ 780e31473dad8bd957c68c494fb858a092c76b
│  │  │  └─ c3a2ff6a1fe6357b0beb003004ccffd1431b57
│  │  ├─ ff
│  │  │  └─ a4d77c254f943fb3115bc4590d53f01b9085ee
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.idx
│  │     └─ pack-5daec5c3ee771efebf94b21149f3f85744b036e1.pack
│  ├─ ORIG_HEAD
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ main
│     │     └─ master
│     ├─ stash
│     └─ tags
├─ .gitignore
├─ 3 CNN 1 Dense 1 Saved Models
├─ api
│  ├─ api.py
│  ├─ best_model.hdf5
│  ├─ model_inference.py
│  ├─ Real 5.jpg
│  ├─ testing.py
│  └─ __init__.py
├─ data
│  ├─ fake_people
│  └─ real_people
├─ deepfake_scraper
│  ├─ chromedriver.exe
│  ├─ data_collection.py
│  ├─ testing.py
│  ├─ webscraping_util.py
│  └─ __init__.py
├─ docker-compose.debug.yml
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfileworking
├─ Dockerfilezzz
├─ environment.yml
├─ environments.yml
├─ LICENSE
├─ Makefile
├─ mentsEXPERIMENT.txt
├─ mentsORIGINAL.txt
├─ model_training.py
├─ myenv.yml
├─ params copy.yaml
├─ params.yaml
├─ Procfile
├─ py36.yml
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ runtime.txt
├─ setup.sh
├─ test
│  ├─ conftest.py
│  ├─ Fake
│  │  └─ test_Fake 4.jpg
│  ├─ test_data_pipeline.py
│  ├─ test_modeling_utils
│  ├─ test_params.yaml
│  └─ __init__.py
├─ utils
│  ├─ custom_metrics_utils.py
│  ├─ data_pipeline_utils.py
│  ├─ modeling_utils.py
│  ├─ plot_metrics_utils.py
│  └─ __init__.py
└─ __init__.py

```