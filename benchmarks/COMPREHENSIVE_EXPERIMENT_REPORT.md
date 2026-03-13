# Comprehensive Research Benchmark Report
Generated on: Sun Mar  8 14:43:33 2026

Total Model Training Optimizers Evaluated: 60
Total HPO Searchers Evaluated: 59

## Best Performing Algorithms by Category
- **bio_inspired**: Best Trainer is SineCosine (Loss: 0.6938)
- **evolutionary**: Best Trainer is CA (Loss: 0.1317)
- **human_based**: Best Trainer is TLBO (Loss: 0.6377)
- **physics**: Best Trainer is FPA (Loss: 0.5135)
- **hybrid**: Best Trainer is Gorilla (Loss: 0.2579)
- **swarm**: Best Trainer is HHO (Loss: 0.1668)

## Detailed Results (Training Weight Optimization)
| Algorithm | Category | Final Loss |
| --- | --- | --- |
| Adam | Gradient | 0.03306229040026665 |
| SGD | Gradient | 0.11553805321455002 |
| CA | evolutionary | 0.13172432780265808 |
| HHO | swarm | 0.1668260097503662 |
| CEM | evolutionary | 0.19946029782295227 |
| PFA | evolutionary | 0.23918215930461884 |
| DFO | swarm | 0.24932102859020233 |
| IGWO | swarm | 0.25059106945991516 |
| Gorilla | hybrid | 0.2579207718372345 |
| PSO | swarm | 0.28558099269866943 |
| DVBA | swarm | 0.3640713095664978 |
| GOA | swarm | 0.4370422065258026 |
| IWOA | swarm | 0.44145575165748596 |
| GA | evolutionary | 0.485897421836853 |
| SSA | swarm | 0.5037004947662354 |
| FPA | physics | 0.5134537816047668 |
| DE | evolutionary | 0.5426557660102844 |
| Clonalg | swarm | 0.5458807945251465 |
| ABCO | swarm | 0.5462132692337036 |
| MBO | swarm | 0.5800028443336487 |
| SMA | hybrid | 0.6034773588180542 |
| SPBO | swarm | 0.6059460639953613 |
| TLBO | human_based | 0.6377352476119995 |
| JSO | hybrid | 0.6397680640220642 |
| Firefly | swarm | 0.6438143849372864 |
| WOA | swarm | 0.6607720255851746 |
| GWO | swarm | 0.6831048727035522 |
| FDA | evolutionary | 0.6868336200714111 |
| PBIL | evolutionary | 0.6931473016738892 |
| Dragonfly | swarm | 0.6934159398078918 |
| SOS | swarm | 0.6935329437255859 |
| SineCosine | bio_inspired | 0.6938325762748718 |
| HUS | swarm | 0.6945794820785522 |
| CSA | swarm | 0.6999183297157288 |
| Salp | swarm | 0.7496839761734009 |
| CatSwarm | hybrid | 0.8413757681846619 |
| DFO2 | swarm | 0.9283076524734497 |
| Bee | swarm | 0.9318098425865173 |
| RandomSearch | swarm | 0.9410384297370911 |
| HarmonySearch | human_based | 0.9655292630195618 |
| ChickenSwarm | hybrid | 1.0976096391677856 |
| Coati | hybrid | 1.1002819538116455 |
| HSA | swarm | 1.1390091180801392 |
| Bat | swarm | 1.146447777748108 |
| ACGWO | swarm | 1.1597617864608765 |
| Cockroach | hybrid | 1.178391456604004 |
| AOA | swarm | 1.220162034034729 |
| Fish | swarm | 1.243270993232727 |
| JY | swarm | 1.2997307777404785 |
| CuckooSearch | swarm | 1.3232818841934204 |
| MVO | bio_inspired | 1.3392367362976074 |
| ARS | evolutionary | 1.3857444524765015 |
| GSA | physics | 1.401357650756836 |
| ALO | bio_inspired | 1.4095244407653809 |
| MFO | bio_inspired | 1.413557529449463 |
| KHA | hybrid | 1.447756290435791 |
| Memetic | swarm | 1.4574025869369507 |
| BBO | bio_inspired | 1.4589112997055054 |
| AFSA | swarm | 1.4861809015274048 |
| EHO | hybrid | 1.6358484029769897 |


## Detailed Results (Hyperparameter Optimization)
| Algorithm | Category | Best Accuracy |
| --- | --- | --- |
| SASearch | physics | 0.9850000143051147 |
| DVBASearch | swarm | 0.9850000143051147 |
| PBILSearch | evolutionary | 0.9800000190734863 |
| AFSASearch | swarm | 0.9800000190734863 |
| JSOSearch | hybrid | 0.9800000190734863 |
| FPASearch | physics | 0.9750000238418579 |
| HHOSearch | swarm | 0.9750000238418579 |
| SSASearch | swarm | 0.9750000238418579 |
| MBOSearch | swarm | 0.9750000238418579 |
| GOASearch | swarm | 0.9700000286102295 |
| CASearch | evolutionary | 0.9700000286102295 |
| PSOSearch | swarm | 0.9700000286102295 |
| AOASearch | swarm | 0.9700000286102295 |
| ChickenSwarmSearch | hybrid | 0.9649999737739563 |
| SMASearch | hybrid | 0.9649999737739563 |
| FDASearch | evolutionary | 0.9649999737739563 |
| GWOSearch | swarm | 0.9649999737739563 |
| ABCOSearch | swarm | 0.9599999785423279 |
| HSASearch | swarm | 0.9599999785423279 |
| KHASearch | hybrid | 0.9599999785423279 |
| CatSwarmSearch | hybrid | 0.9599999785423279 |
| TLBOSearch | human_based | 0.9599999785423279 |
| GSASearch | physics | 0.9599999785423279 |
| MFOSearch | bio_inspired | 0.9599999785423279 |
| ARSSearch | evolutionary | 0.9599999785423279 |
| SineCosineSearch | bio_inspired | 0.9599999785423279 |
| PFASearch | evolutionary | 0.9599999785423279 |
| CuckooSearchHT | swarm | 0.9599999785423279 |
| IGWOSearch | swarm | 0.9599999785423279 |
| ACGWOSearch | swarm | 0.9599999785423279 |
| CEMSearch | evolutionary | 0.9549999833106995 |
| MemeticSearch | swarm | 0.9549999833106995 |
| JYSearch | swarm | 0.9549999833106995 |
| BeeSearch | swarm | 0.9549999833106995 |
| CoatiSearch | hybrid | 0.9549999833106995 |
| EHOSearch | hybrid | 0.9549999833106995 |
| DESearch | evolutionary | 0.9549999833106995 |
| FireflySearch | swarm | 0.9549999833106995 |
| HUSSearch | swarm | 0.949999988079071 |
| CSASearch | swarm | 0.949999988079071 |
| CockroachSearch | hybrid | 0.949999988079071 |
| HarmonySearchHT | human_based | 0.949999988079071 |
| RandomSearchHT | swarm | 0.949999988079071 |
| ClonalgSearch | swarm | 0.949999988079071 |
| BatSearch | swarm | 0.949999988079071 |
| DFOSearch | swarm | 0.949999988079071 |
| SOSSearch | swarm | 0.949999988079071 |
| ALOSearch | bio_inspired | 0.9449999928474426 |
| SalpSearch | swarm | 0.9449999928474426 |
| WOASearch | swarm | 0.9449999928474426 |
| BBOSearch | bio_inspired | 0.9449999928474426 |
| GorillaSearch | hybrid | 0.9399999976158142 |
| GASearch | evolutionary | 0.9399999976158142 |
| MVOSearch | bio_inspired | 0.9399999976158142 |
| DragonflySearch | swarm | 0.9399999976158142 |
| FishSearch | swarm | 0.9399999976158142 |
| IWOASearch | swarm | 0.9300000071525574 |
| DFO2Search | swarm | 0.925000011920929 |
| SPBOSearch | swarm | 0.9150000214576721 |
