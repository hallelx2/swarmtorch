# Comprehensive Research Benchmark Report
Generated on: Tue Mar 10 10:43:42 2026

Total Model Training Optimizers Evaluated: 60
Total HPO Searchers Evaluated: 59

## Best Performing Algorithms by Category
- **bio_inspired**: Best Trainer is SineCosine (Loss: 0.7708)
- **evolutionary**: Best Trainer is CA (Loss: 0.2187)
- **human_based**: Best Trainer is TLBO (Loss: 0.6901)
- **physics**: Best Trainer is FPA (Loss: 0.7203)
- **hybrid**: Best Trainer is Gorilla (Loss: 0.2728)
- **swarm**: Best Trainer is IWOA (Loss: 0.2400)

## Detailed Results (Training Weight Optimization)
| Algorithm | Category | Final Loss |
| --- | --- | --- |
| Adam | Gradient | 0.027349941432476044 |
| SGD | Gradient | 0.12745100259780884 |
| CA | evolutionary | 0.21874654293060303 |
| PFA | evolutionary | 0.22179865837097168 |
| CEM | evolutionary | 0.22524750232696533 |
| IWOA | swarm | 0.2399510145187378 |
| Gorilla | hybrid | 0.27279114723205566 |
| GOA | swarm | 0.28064194321632385 |
| DFO | swarm | 0.29363566637039185 |
| HHO | swarm | 0.33173149824142456 |
| IGWO | swarm | 0.35967591404914856 |
| PSO | swarm | 0.36392876505851746 |
| DVBA | swarm | 0.38466307520866394 |
| Clonalg | swarm | 0.46932533383369446 |
| Firefly | swarm | 0.47261345386505127 |
| FDA | evolutionary | 0.48151740431785583 |
| DE | evolutionary | 0.493569016456604 |
| GA | evolutionary | 0.5120543837547302 |
| SSA | swarm | 0.5476040840148926 |
| ABCO | swarm | 0.5686624646186829 |
| SMA | hybrid | 0.62315833568573 |
| Salp | swarm | 0.6289653182029724 |
| SPBO | swarm | 0.6366990804672241 |
| HUS | swarm | 0.6404846906661987 |
| JSO | hybrid | 0.6434437036514282 |
| GWO | swarm | 0.6477985382080078 |
| MBO | swarm | 0.6646429896354675 |
| SOS | swarm | 0.6796358227729797 |
| WOA | swarm | 0.6832001209259033 |
| Dragonfly | swarm | 0.6852114796638489 |
| TLBO | human_based | 0.6900981068611145 |
| CSA | swarm | 0.690549910068512 |
| PBIL | evolutionary | 0.6931473016738892 |
| Bee | swarm | 0.7153347134590149 |
| FPA | physics | 0.7202872633934021 |
| SineCosine | bio_inspired | 0.7708128094673157 |
| CatSwarm | hybrid | 0.7822293639183044 |
| RandomSearch | swarm | 0.9576318264007568 |
| HarmonySearch | human_based | 0.9678534865379333 |
| BBO | bio_inspired | 1.122300624847412 |
| CuckooSearch | swarm | 1.1293624639511108 |
| Fish | swarm | 1.156600832939148 |
| HSA | swarm | 1.1732394695281982 |
| ALO | bio_inspired | 1.2519598007202148 |
| AFSA | swarm | 1.3024922609329224 |
| Bat | swarm | 1.3055055141448975 |
| Cockroach | hybrid | 1.3235371112823486 |
| DFO2 | swarm | 1.3432682752609253 |
| EHO | hybrid | 1.3501793146133423 |
| ACGWO | swarm | 1.3662052154541016 |
| Memetic | swarm | 1.3912653923034668 |
| JY | swarm | 1.3927199840545654 |
| KHA | hybrid | 1.3947255611419678 |
| MVO | bio_inspired | 1.4407864809036255 |
| AOA | swarm | 1.4603041410446167 |
| GSA | physics | 1.4768476486206055 |
| ARS | evolutionary | 1.5305553674697876 |
| Coati | hybrid | 1.5556957721710205 |
| ChickenSwarm | hybrid | 1.5732109546661377 |
| MFO | bio_inspired | 1.7682843208312988 |


## Detailed Results (Hyperparameter Optimization)
| Algorithm | Category | Best Accuracy |
| --- | --- | --- |
| PFASearch | evolutionary | 0.9549999833106995 |
| HarmonySearchHT | human_based | 0.9549999833106995 |
| PBILSearch | evolutionary | 0.9549999833106995 |
| GSASearch | physics | 0.9549999833106995 |
| JYSearch | swarm | 0.9549999833106995 |
| FDASearch | evolutionary | 0.949999988079071 |
| CockroachSearch | hybrid | 0.949999988079071 |
| FishSearch | swarm | 0.949999988079071 |
| GOASearch | swarm | 0.949999988079071 |
| WOASearch | swarm | 0.949999988079071 |
| CatSwarmSearch | hybrid | 0.949999988079071 |
| ClonalgSearch | swarm | 0.949999988079071 |
| ACGWOSearch | swarm | 0.949999988079071 |
| CoatiSearch | hybrid | 0.9449999928474426 |
| AFSASearch | swarm | 0.9449999928474426 |
| IGWOSearch | swarm | 0.9449999928474426 |
| IWOASearch | swarm | 0.9449999928474426 |
| RandomSearchHT | swarm | 0.9449999928474426 |
| ABCOSearch | swarm | 0.9449999928474426 |
| HHOSearch | swarm | 0.9449999928474426 |
| DragonflySearch | swarm | 0.9449999928474426 |
| SSASearch | swarm | 0.9449999928474426 |
| MBOSearch | swarm | 0.9449999928474426 |
| ARSSearch | evolutionary | 0.9449999928474426 |
| GASearch | evolutionary | 0.9449999928474426 |
| JSOSearch | hybrid | 0.9449999928474426 |
| SineCosineSearch | bio_inspired | 0.9449999928474426 |
| MemeticSearch | swarm | 0.9449999928474426 |
| GWOSearch | swarm | 0.9399999976158142 |
| DVBASearch | swarm | 0.9399999976158142 |
| FireflySearch | swarm | 0.9399999976158142 |
| HSASearch | swarm | 0.9399999976158142 |
| SalpSearch | swarm | 0.9399999976158142 |
| CuckooSearchHT | swarm | 0.9399999976158142 |
| SOSSearch | swarm | 0.9399999976158142 |
| CASearch | evolutionary | 0.9399999976158142 |
| MFOSearch | bio_inspired | 0.9399999976158142 |
| ALOSearch | bio_inspired | 0.9399999976158142 |
| BBOSearch | bio_inspired | 0.9399999976158142 |
| DESearch | evolutionary | 0.9399999976158142 |
| FPASearch | physics | 0.9399999976158142 |
| CEMSearch | evolutionary | 0.9399999976158142 |
| EHOSearch | hybrid | 0.9399999976158142 |
| BeeSearch | swarm | 0.9399999976158142 |
| BatSearch | swarm | 0.9350000023841858 |
| SMASearch | hybrid | 0.9350000023841858 |
| KHASearch | hybrid | 0.9350000023841858 |
| TLBOSearch | human_based | 0.9350000023841858 |
| ChickenSwarmSearch | hybrid | 0.9350000023841858 |
| MVOSearch | bio_inspired | 0.9350000023841858 |
| GorillaSearch | hybrid | 0.9350000023841858 |
| DFO2Search | swarm | 0.9350000023841858 |
| HUSSearch | swarm | 0.9350000023841858 |
| DFOSearch | swarm | 0.9350000023841858 |
| PSOSearch | swarm | 0.9300000071525574 |
| SPBOSearch | swarm | 0.9300000071525574 |
| AOASearch | swarm | 0.9300000071525574 |
| SASearch | physics | 0.925000011920929 |
| CSASearch | swarm | 0.925000011920929 |
