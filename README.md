
#### A brief survey  : AI in Risk Management and Insurance

---------------

> #### Contents

- [Misc](#misc)
- [Finance Industry](#finance-industry)
- [Finance Technology](#finance-technology)
- [AI in Risk Management](#ai-in-risk-management)
- [AI in Insurance](#ai-in-insurance)
- [Literature review](#literature-review)
- [Startups](#startups)
- [Appendix](#appendix)


[Back to top](#contents)


--------------

#### Misc

[Back to top](#contents)

What's the future looks like for DL & Finance?
* Igor Halperin: RL on management [quora](https://www.quora.com/What-would-be-the-next-big-thing-in-quantitative-finance-after-machine-learning-big-data-high-frequency-algorithmic-trading-How-is-the-industry-going-to-evolve-in-the-next-10-years)
* Igor Halperin: deep learning is useful for mortgage risk, as was shown by Kay Giesecke and his group at Stanford, as well as for credit scoring/credit cards, P2P lending [quora](https://www.quora.com/Is-deep-learning-useful-in-finance-outside-of-high-frequency-trading)
* Igor Halperin: RL on credit scoring: [Quora](https://www.quora.com/Which-books-or-lecture-notes-should-I-learn-to-begin-develop-reinforcement-learning-models-for-credit-scoring)

Finance data property w.r.t. Deep Learning

* Too noisy to use ML model not stationary
* Data are small-to-medium, where deep learning needs large data
* Igor Halperin: (Deep) feedforward neural nets, autoencoders and LSTM seem to be the state of the art of ML in finance, based on what is published and presented at different conferences. [Quora](https://www.quora.com/What-is-the-state-of-the-art-in-simulating-financial-price-series)
* Igor Halperin: ML currently un-explainable in finance [Quora](https://www.quora.com/What-is-being-done-at-the-academic-level-so-as-to-make-ML-algorithms-more-interpretable-for-regulators-in-finance-risk-space)
* Financial time-series is a partial information game (POMDP), so needs RL
> * High frequency trading and algorithmic trading are the main drivers of price at short intervals (< 1 day).
> * Opening and closing prices have their own patterns - both in stocks and futures - the two asset classes I have worked with.
> * News and rumors are the driving forces when it comes to multi-day horizons. Specific company news can happen at any time without any prior notice. However, the timeline for some events is known beforehand. Company result schedule, as well as the economic data calendar, are known beforehand.
> * Value investing and economic cycles matter the most when it comes to price changes at a multi-year range.



------------

#### Finance Industry

[Back to top](#contents)



------------------

#### Finance Technology

[Back to top](#contents)




------------------

#### AI in Risk Management

[Back to top](#contents)

Financial Risk Management Categories
1. Market Risk
2. Credit Risk
3. Liquidity Risk
4. Operational Risk

Casualty Actuarial Society (CAS) define Enterprise risk management (ERM)
1. Hazard risk: Liability torts, Property damage, Natural catastrophe
2. Financial risk: Pricing risk, Asset risk, Currency risk, Liquidity risk
3. Operational risk: Customer satisfaction, Product failure, Integrity, Reputational risk; Internal Poaching; Knowledge drain
4. Strategic risks: Competition, Social trend, Capital availability


[GLM] Statistical Method can be classified by this
* Tree-based methods (e.g. Random Forest, AdaBoost, Gradient Boost Machine);
* Kernel-based methods (e.g. Support Vector Machine, Kernel Learning);
* Neuron-based methods (e.g. Neural Network, "Deep Learning" or Restricted Boltzmann Machine);
* Graph/network-based methods (e.g. Naïve Bayes, Hidden Markov Model, Deep Belief Network).

Insurance moves from reactive to predictive 
- not just using data but also create data
1. Underwriting: rating(Frequency and Severity)
- CNN: Identify Roof Type
- Flight delay insurance
- Lemonade Insurance
- Car Insurance chips
- Healthcare insurance: Medical Imaging eg. Google Deepmind AI detect eye disease [two mins paper's video](https://www.youtube.com/watch?v=wR2OlsF1CEY)
- Q: legistitive allow ratemaking? on automobile??
2. Underwriting: Fraud Detection
- Credit card fraud detection is successful, will that be able to apply to insurance?
3. Underwriting: Customer segmentation
-----------------------------------------
4. Marketing: Personalized marketing
5. Marketing: Recommendation engines
-----------------------------------------
6. Claims prediction


make a small twist from stat to ML/DL
- example 想一下有什麼例子概念是一樣的，但是小小的轉變一下就可以變很厲害 搜集資料？ lemonade?
- 模型方面



- Deep Learning’s Killer App for Finance? [Article](https://aldousbirchall.com/2017/03/25/deep-learnings-killer-app/)





------------------

#### AI in Insurance

[Back to top](#contents)
SOA: 2019Jan: [Machine-Learning Methods for Insurance Applications-A Survey](https://www.soa.org/resources/research-reports/2019/machine-learning-methods/)
* Machine-Learning Methods for Insurance Applications
* Group Long-Term Disability Jupyter File
* Long-Term Care Jupyter File
* [Group Long-Term Disability Data Set Comparison](file:///Users/Yu-Ying/Downloads/group-ltd-data-set.html)
* Long-Term Care Data Set Comparison

[ratemaking]
- 作者有Bengio!!: Statistical Learning Algorithms Applied to Automobile Insurance Ratemaking [paper](https://www.iro.umontreal.ca/~vincentp/Publications/itii.pdf) [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Statistical%20Learning%20Algorithms%20Applied%20to%20Automobile%20Insurance%20Ratemaking.md)


[parameter estimation]
- [ML-unsupervised?] A MACHINE-LEARNING APPROACH TO PARAMETER ESTIMATION [article](https://www.casact.org/pubs/monographs/papers/06-Kunce-Chatterjee.pdf) 



[Claims]
- R/Shiny - Machine Learning for insurance claims [Video](https://www.youtube.com/watch?v=3KNlT3evDkc) [Article](https://www.r-bloggers.com/machine-learning-for-insurance-claims/) [More Shiny](https://www.youtube.com/channel/UC4cF877tXYUa3h3Ou6tcxXA/videos)

[Fraud Detection] Casualty Insurance face lots of fraud.

- Top 10 Data Science Use Cases in Insurance [Article](https://medium.com/activewizards-machine-learning-company/top-10-data-science-use-cases-in-insurance-8cade8a13ee1)
- 2018.07 AI IN P&C INSURANCE. Pragmatic Approaches for Today, Promise for Tomorrow. [reviews]() [pdf](https://capeanalytics.com/wp-content/uploads/2018/11/SMA-AI-in-PC-Insurance-2018.pdf) 



[loss]
- [GRU] 2018 Actuarial Applications of Deep Learning_Loss Reserving and Beyond [slide](https://ibnr.netlify.com/#29) / [arxiv](https://arxiv.org/pdf/1804.09253.pdf) / [DeepTriangle_github_R](https://github.com/kevinykuo/deeptriangle)

- KPMG: Learning to trust your digital actuary: New technologies can automate loss reserve analysis, providing insurers with more timely data and deeper insights [report](https://assets.kpmg/content/dam/kpmg/pdf/2016/07/learning-to-trust-your-digital-actuary.pdf)


[Company]
- Allianz's 
> bot Allie, an online assistant available 24/7 to answer personal lines customers’ questions. It is also using machine learning to carry out risk assessments and support automated underwriting in the small- to medium-enterprise (SME) space.


[Decentralize Insurance] \
Decentralized Insurance Developer Conference 2017 [Playlist](https://www.youtube.com/playlist?list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO) (Bold is something I'm more interested in.) 
- **Keynote: Decentralized Decision Making** [Video](https://www.youtube.com/watch?v=-j6Esc_Pm_I&index=2&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO)
- Workshop: How to Develop a Blockchain-based Insurance Solution in a Few Hours [Video](https://www.youtube.com/watch?v=OqlThIN-kC4&index=3&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO)
- Bringing the True P2P Mutual Back to Insurance [Video](https://www.youtube.com/watch?v=-0kEd9pcllc&index=4&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO)
- Insurance and Blockchain, Show Me the Code! [Video](https://www.youtube.com/watch?v=7cHvg6KugcE&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=5)
- **Workshop: Smart Contract Car Insurance in 30 min** [Video](https://www.youtube.com/watch?v=paWuuz8pZQw&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=7)
- Workshop - How to Build Decentralized Insurance Apps [Video](https://www.youtube.com/watch?v=Wlii-juuuEY&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=8)
- Fireside Chat - View on Decentralized Insurance From an Insurance VC [Video](https://www.youtube.com/watch?v=tiZYSAWW49w&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=9)
- Feeding Authentic Data into Blockchain: Insurance Dapps on Steroids [Video](https://www.youtube.com/watch?v=zLJiWL3VS6I&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=10)
- **A Simple Probability Model For Decentralized Insurance** [Video](https://www.youtube.com/watch?v=NsF0mOvyDp8&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=11)
- Standardized Protocols For Decentralized Insurance [Video](https://www.youtube.com/watch?v=y3_a7iwWldI&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=12)
- **The Value Behind Hyperledger** [Video](https://www.youtube.com/watch?v=2OaAeWke7LY&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=13)
- **Introduction To Machine Learning For Insurance Use Cases** [Video](https://www.youtube.com/watch?v=vq6G7NnWlDQ&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=14)
- **Fireside Chat: Prediction Markets For Insurance** [Video](https://www.youtube.com/watch?v=dOrLEuD3Arw&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=15)
- **Blockchain & Insurance | Regulation & Compliance** [Video](https://www.youtube.com/watch?v=Ll-1wS3pcSQ&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=16)
- Risk Pool Tokens [Video](https://www.youtube.com/watch?v=Rvz2RjlGFl4&list=PLv_C3XL8vCF4TfX9UKZgA8wWZQuvuqFyO&index=17)

Decentralized Insurance Developer Conference 2018 [Playlist](https://www.youtube.com/playlist?list=PLv_C3XL8vCF4rniMwrZtaLTFNzCrGT1FF)


----------

#### Literature review


[Back to top](#contents)

BirdView for Risk Management:
- AI and Machine Learning for Risk Management [paper](https://www.garp.org/#!/risk-intelligence/all/all/a1Z1W000004AoB1UAK)
- Accenture(consulting): Validating Machine Learning and AI Models in Financial [report](https://www.accenture.com/t20180427T082714Z__w__/us-en/_acnmedia/Accenture/Conversion-Assets/MainPages/Documents/Global/Accenture-Emerging-Trends-in-the-Validation-of-ML-and-AI-Models.pdf)

Survival Analysis:(relatively mature)
- [ML] 2017: Machine Learning for Survival Analysis: A Survey [paper](https://arxiv.org/pdf/1708.04649.pdf)
- [repository](https://github.com/robi56/Survival-Analysis-using-Deep-Learning)
- [DeepSurv package](https://github.com/jaredleekatzman/DeepSurv)
- [SurvivalNet package](https://github.com/CancerDataScience/SurvivalNet)


Market Risk Manegement:
- [LSTM+stacked autoencoder] 2017: A deep learning framework for financial time series using stacked autoencoders and long-short term memory [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5510866/)
- [GANs] Using Bidirectional Generative Adversarial Networks to estimate Value-at-Risk for Market Risk Management [Medium](https://towardsdatascience.com/using-bidirectional-generative-adversarial-networks-to-estimate-value-at-risk-for-market-risk-c3dffbbde8dd) [Github](https://github.com/hamaadshah/market_risk_gan_keras)
- [GAN] 2005/Two-Step Disentanglement for Financial Data [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Two-Step%20Disentanglement%20for%20Financial%20Data.md) [paper](https://arxiv.org/pdf/1709.00199.pdf)
- [GAN/LSTM/CNN] Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets [paper](https://www.hindawi.com/journals/mpe/2018/4907423/)
- [RL] RISK-AVERSE DISTRIBUTIONAL REINFORCEMENT LEARNING: a cvar optimization approach [master thesis](https://dspace.cvut.cz/bitstream/handle/10467/76432/F3-DP-2018-Stanko-Silvestr-thesis.pdf)

Mortgage Risk
- [DL] 2015: Deep Learning for Mortgage Risk [paper](https://stanford.app.box.com/s/0iqyz2zt82uvqjb5cn8tskxxe7m0i0zo)

Credit Risk
- [DL] 2018: Credit Risk Analysis Using Machine and Deep Learning Models [paper](https://halshs.archives-ouvertes.fr/halshs-01835164/document)
- [LSTM] 2018: Deep Credit Risk Ranking with LSTM [talk+slide](https://databricks.com/session/productionizing-credit-risk-analytics-with-lstm-tensorspark-a-wells-fargo-case-study) 
- [DL] Credit Card Default Prediction Using TensorFlow (Part-1 Deep Neural Networks) [medium](https://medium.com/@Saadism/credit-card-default-prediction-using-tensorflow-part-1-deep-neural-networks-ef22cfd4d278)
- [ML] 2018: Ensemble Learning or Deep Learning? Application to Default risk analysis. [student's paper in TW data](http://www.econ.kobe-u.ac.jp/RePEc/koe/wpaper/2018/1802.pdf)
- [ML] 2017: ANALYSIS OF FINANCIAL CREDIT RISK USING MACHINE LEARNING [thesis](https://arxiv.org/pdf/1802.05326.pdf)

Option Pricing:
- [RL] 2018 Igor Halperin: Model-Free Option Pricing with Reinforcement Learning [slide](https://cfe.columbia.edu/files/seasieor/industrial-engineering-operations-research/IgorHalperin_Columbia_ML_in_finance_QLBS.pdf) / [coursera](https://www.coursera.org/specializations/machine-learning-reinforcement-finance)
- [RL] 2017 Igor Halperin: QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds [paper](https://arxiv.org/pdf/1712.04609v2.pdf)
- [RL] 2017 Igor Halperin: Inverse Reinforcement Learning for Marketing [paper](https://arxiv.org/pdf/1712.04612v1.pdf)
- [RL] 2018 Igor Halperin: The QLBS Q-Learner Goes NuQLear: Fitted Q Iteration, Inverse RL, and Option Portfolios [paper](https://arxiv.org/pdf/1801.06077v1.pdf)
- [RL] 2018 Igor Halperin: Market Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand Inference with Free Energy (or, How We Learned to Stop Worrying and Love Bounded Rationality) [paper](https://arxiv.org/pdf/1805.06126v1.pdf)
- [RL] 2018: Pricing Options with an Artificial Neural Network: A Reinforcement Learning Approach [thesis w/ simplified python](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2565055/Masteroppgave%20HaakonTrønnes.g.pdf?sequence=1&isAllowed=y) / [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Pricing%20Options%20with%20an%20Artificial%20Neural%20Network:%20A%20Reinforcement%20Learning%20Approach.md)
- [NN] 2018 World Quant Blog: BEYOND BLACK-SCHOLES: A NEW OPTION FOR OPTIONS PRICING [blog](https://www.weareworldquant.com/en/thought-leadership/beyond-black-scholes-a-new-option-for-options-pricing/)
- [DNN] 2018: The Day my Computer Won the Nobel Prize (Neural Network Option Pricing) [medium w/ python](https://medium.com/datadriveninvestor/the-day-my-computer-won-the-nobel-prize-neural-network-option-pricing-d29b4379f1d2)
- [DNN] 2018: Deeply Learning Derivatives [paper](https://arxiv.org/abs/1809.02233) / [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Deeply%20Learning%20Derivatives.md)
- [GNN] 2017: Gated Neural Networks for Option Pricing: Rationality by Design [paper](https://arxiv.org/pdf/1609.07472.pdf) / [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Gated%20Neural%20Networks%20for%20Option%20Pricing:%20Rationality%20by%20Design.md) / [github?](https://github.com/arraystream/fftoptionlib) / [blog](https://www.arraystream.com/2017/05/08/fftoptionlib/)
- [FeedFroward] 2017: Machine Learning in Finance: The Case of Deep Learning for Option Pricing [paper](https://srdas.github.io/Papers/BlackScholesNN.pdf) /  [pythob code](https://srdas.github.io/DLBook/DeepLearningWithPython.html#option-pricing) / [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Machine%20Learning%20in%20Finance:%20The%20Case%20of%20Deep%20Learning%20for%20Option%20Pricing.md)
- [NN] 2012 An Option Pricing Model That Combines Neural Network Approach and Black Scholes Formula
- [ANN] 2013: Option Pricing Using Artificial Neural Networks: An Australian Perspective [dissertation](https://pure.bond.edu.au/ws/portalfiles/portal/18243185/Option_Pricing_Using_Artificial_Neural_Networks.pdf) / [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/Option%20Pricing%20Using%20Artificial%20Neural%20Networks:%20An%20Australian%20Perspective.md)

Pricing:
- [DNN] 2018: The Day I Taught My Computer What Duration Is (Neural Network Bond Pricing) [blog w/ code](https://www.linkedin.com/pulse/day-i-taught-my-computer-what-duration-neural-network-jacob-bourne/)
- [ML] Machine Learning Methods to Perform Pricing Optimization. A Comparison with Standard GLMs. [paper](https://www.variancejournal.org/articlespress/articles/Machine-Spedicato.pdf)
- [ML] Tree-based machine learning for insurance pricing. [slide](https://kuleuvencongres.be/eaj2018/documents/presentations/1-roel-henckaerts.pdf)
- [ML] Machine Learning application in non-life pricing. Frequency modelling: an educational case study. [report](https://reacfin.com/en/sites/default/files/documents/20170914%20Machine%20Learning%20applications%20for%20non-life%20pricing.pdf) 
- [ML] Data Analytics for Non-Life Insurance Pricing [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2870308)
- [ML] Data Science in Non-Life Insurance Pricing. Predicting Claims Frequencies using Tree-Based Models. [Master Thesis](https://www.ethz.ch/content/dam/ethz/special-interest/math/imsf-dam/documents/walter-saxer-preis/ma-zoechbauer.pdf)
- [ML] Overview and practical application of Machine learning in Pricing [report](https://www.casact.org/education/spring/2017/presentations/C-24.pdf) conclusion: domain knowledge win


Portfolio Management
- [AttentionRNN] 2017: A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction (The Nonlinear autoregressive exogenous (NARX) model) [paper](https://arxiv.org/abs/1704.02971) / [Pytorch example blog](http://chandlerzuo.github.io/blog/2017/11/darnn) / [update Github](https://github.com/Seanny123/da-rnn) / [note]()
- [RL] 2015: A Comprehensive Survey on Safe Reinforcement Learning [paper](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) [note](https://github.com/yingpublic/RMI-AI/blob/master/review/academia/A%20Comprehensive%20Survey%20on%20Safe%20Reinforcement%20Learning.md)
- [RL] 2005/J of AI: Risk-Sensitive Reinforcement Learning Applied to Control under Constraints [paper](https://arxiv.org/pdf/1109.2147.pdf) [note]()
- [RL-deepQ] Portfolio Management using Reinforcement Learning [paper](http://cs229.stanford.edu/proj2016/report/JinElSaawy-PortfolioManagementusingReinforcementLearning-report.pdf) [note]()
- [RL] 1996/ Use Of Neural Network Ensembles for Portfolio Selection and Risk Management


===
- FinBrain: When Finance Meets AI 2.0. [arxiv](https://arxiv.org/abs/1808.08497v1)
- Deep Learning in Finance. [arxiv](https://arxiv.org/abs/1602.06561v3) - 2018
- Forecasting Economics and Financial Time Series: ARIMA vs. LSTM. [arxiv](https://arxiv.org/abs/1803.06386v1) - 2018
- TSViz: Demystification of Deep Learning Models for Time-Series Analysis. [arxiv](https://arxiv.org/abs/1802.02952v1) - 2018
- Geometric Learning and Filtering in Finance. [arxiv](https://arxiv.org/abs/1710.05829v2) - 2017
- On Feature Reduction using Deep Learning for Trend Prediction in Finance. [arxiv](https://arxiv.org/abs/1704.03205v1) - 2017
- Forecasting Volatility in Indian Stock Market using Artificial Neural Network with Multiple Inputs and Outputs. [arxiv](https://arxiv.org/abs/1604.05008v1) - 2016
- Financial Market Modeling with Quantum Neural Networks. [arxiv](https://arxiv.org/abs/1508.06586v1) - 2015
- Identifying Metaphoric Antonyms in a Corpus Analysis of Finance Articles. [arxiv](https://arxiv.org/abs/1212.3139v2) - 2013
- Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. [arxiv](https://arxiv.org/abs/1307.5336v2) - 2013
- Identifying Metaphor Hierarchies in a Corpus Analysis of Finance Articles. [arxiv](https://arxiv.org/abs/1212.3138v1) - 2012


Fraud detection:

- [One-Class Adversarial Nets for Fraud Detection](https://arxiv.org/pdf/1803.01798.pdf)
- [Detection of Anomalies in Large-Scale
Accounting Data using Deep Autoencoder
Networks](https://arxiv.org/pdf/1709.05254.pdf)
- [Bypass Fraud Detection:
Artificial Intelligence Approach](https://arxiv.org/ftp/arxiv/papers/1711/1711.04627.pdf)
- [Transaction Fraud Detection Using GRU-centered Sandwich-structured Model](https://arxiv.org/ftp/arxiv/papers/1711/1711.01434.pdf)
- [Opinion Fraud Detection via Neural Autoencoder Decision Forest](https://arxiv.org/pdf/1805.03379.pdf)
- [Instance-Level Explanations for Fraud Detection: A Case Study](https://arxiv.org/abs/1806.07129)
- [A Comprehensive Survey of
Data Mining-based Fraud Detection Research](https://arxiv.org/ftp/arxiv/papers/1009/1009.6119.pdf)
- [Streaming Active Learning Strategies for Real-Life Credit Card Fraud Detection: Assessment and Visualization](https://arxiv.org/abs/1804.07481)
- [Sequential Behavioral Data Processing Using Deep Learning and the Markov Transition Field in Online Fraud Detection](https://arxiv.org/abs/1808.05329v1)
- [A Survey of Credit Card Fraud Detection Techniques: Data and Technique Oriented Perspective](https://arxiv.org/abs/1611.06439v1)


Credit scoring:

- [Segment-Based Credit Scoring Using Latent Clusters in the Variational Autoencoder](https://arxiv.org/abs/1806.02538v1)
- [Improving a Credit Scoring Model by Incorporating Bank Statement Derived Features](https://arxiv.org/abs/1611.00252v2)
- [Transfer Learning Using Logistic Regression in Credit Scoring](https://arxiv.org/abs/1212.6167v1)

PF:

- [Different but Equal: Comparing User Collaboration with Digital Personal Assistants vs. Teams of Expert Agents](https://arxiv.org/abs/1808.08157v1)
- [PersonaBank: A Corpus of Personal Narratives and Their Story Intention Graphs](https://arxiv.org/abs/1708.09082v1)
- [Intelligent Personal Assistant with Knowledge Navigation](https://arxiv.org/abs/1704.08950v1)

Sentiment Analysis:

- [Deep Learning for Sentiment Analysis : A Survey](https://arxiv.org/abs/1801.07883v2)
- [Financial Aspect-Based Sentiment Analysis using Deep Representations](https://arxiv.org/abs/1808.07931v1)
- [The Evolution of Sentiment Analysis - A Review of Research Topics, Venues, and Top Cited Papers](https://arxiv.org/abs/1612.01556v4)

----------

#### Libraries

- [ml-fraud-detection](https://github.com/georgymh/ml-fraud-detection)
- [watson-banking-chatbot](https://github.com/IBM/watson-banking-chatbot)


![architecture](https://github.com/IBM/watson-banking-chatbot/blob/master/doc/source/images/architecture.png)

- [Deep-Trading](https://github.com/Rachnog/Deep-Trading)
- [Trading-Gym](https://github.com/thedimlebowski/Trading-Gym)
- [gym-trading](https://github.com/hackthemarket/gym-trading)
- [mosquito](https://github.com/miro-ka/mosquito)
- [deep-algotrading](https://github.com/LiamConnell/deep-algotrading)
- [Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras)
- [Fraud-detection-using-deep-learning](https://github.com/aaxwaz/Fraud-detection-using-deep-learning)
- [credit-card-fraud](https://github.com/ellisvalentiner/credit-card-fraud)
- [ml-fraud-detection](https://github.com/georgymh/ml-fraud-detection)

--------------

#### Startups

[Back to top](#contents)



-------------

#### Appendix

[Back to top](#contents)

![AI-in-FinTech-Market-Map-Image3](https://cbi-blog.s3.amazonaws.com/blog/wp-content/uploads/2017/03/AI-in-FinTech-Market-Map-Image3.png)


![Screen-Shot-2018-07-18-at-7.38.18-PM-1024x467](https://s3.amazonaws.com/cbi-research-portal-uploads/2018/07/18193851/Screen-Shot-2018-07-18-at-7.38.18-PM-1024x467.png)



![Artificial-Intelligence-Financial-Industry](https://thefinancialbrand.com/wp-content/uploads/2018/05/Artificial-Intelligence-Financial-Industry.png)
