# import the necessary packages
from assets import _config
from assets.datapipeline import get_dataloader 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision import transforms


# initialize test transform pipeline
testTransform = Compose([
    Resize((_config.IMAGE_SIZE, _config.IMAGE_SIZE)),
    ToTensor(),
    Normalize(mean=_config.MEAN, std=_config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(_config.MEAN, _config.STD)]
invStd = [1/s for s in _config.STD]


# define our denormalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create the test dataset
trainDataset = ImageFolder(_config.TRAIN_PATH, testTransform)

# initialize the test data loader
trainLoader = get_dataloader(trainDataset, _config.PRED_BATCH_SIZE)

#print(len(testLoader.dataset.classes))

MEANING = {
    "Adinkrahene": "greatness, charisma, leadership", 
    "Akoben": "vigilance, wariness", 
    "Akofena": "courage, valor", 
    "Akoko Nan": "mercy, nurturing", 
    "Akoma": "patience & tolerance", 
    "Akoma Ntoaso": "understanding, agreement", 
    "Ananse Ntontan": "wisdom, creativity", 
    "Asase Ye Duru": "divinity of Mother Earth", 
    "Aya": "endurance, resourcefulness",
    "Bese Saka": "affluence, abundance, unity",
    "Bi Nka Bi": "peace, harmony", 
    "Boa me na me mmoa wo": "cooperation, interdependence", 
    "Dame dame": "intelligence, ingenuity", 
    "Denkyem": "adaptability", 
    "Duafe": "beauty, hygiene, feminine qualities", 
    "Dwennimmen": "humility and strength", 
    "Eban": "love, safety, security", 
    "Epa": "law, justice, slavery", 
    "Ese ne tekrema": "friendship, interdependence",
    "Fawohodie": "independence, freedom, emancipation",
    "Fihankra": "security, safety", 
    "Fofo": "jealousy, envy", 
    "Funtunfunefu-denkyemfunefu": "democracy, unity in diversity", 
    "Gye Nyame": "supremacy of God", 
    "Hwemudua": "examination, quality control", 
    "Hye wonhye": "imperishability, endurance", 
    "Kete pa": "good marriage", 
    "Kintinkantan": "arrogance, extravagance", 
    "Mate masie": "wisdom, knowledge, prudence",
    "Me ware wo": "commitment, perseverance",
    "Mframadan": "fortitude, preparedness", 
    "Mmere dane": "change, life's dynamics", 
    "Mmusuyidee": "good fortune, sanctity", 
    "Mpatapo": "peacemaking, reconciliation", 
    "Mpuannum": "priestly office, loyalty, adroitness", 
    "Nea Ope Se Obedi Hene": "service, leadership", 
    "Nea onnim no sua a, ohu": "knowledge, life-long education", 
    "Nkonsonkonson": "unity, human relations", 
    "Nkyimu": "skillfulness, precision",
    "Nkyinkyim": "initiative, dynamism, versatility",
    "Nsaa": "excellence, genuineness, authenticity", 
    "Nsoromma": "guardianship", 
    "Nyame  Nti": "faith & trust in God", 
    "Nyame Nnwu Na Mawu": "life after death", 
    "Nyame biribi wo soro": "hope", 
    "Nyame dua": "God's protection and presence", 
    "Nyame ye Ohene": "God is King", 
    "Nyansapo": "wisdom, ingenuity, intelligence and patience", 
    "Odo Nnyew Fie Kwan": "power of love",
    "Okodee Mmowere": "bravery, strength",
    "Onyankopon Adom Nti Biribira Beye Yie": "hope, providence, faith", 
    "Osram ne Nsoromma": "love, faithfulness, harmony", 
    "Owo Foro Adobe": "steadfastness, prudence, diligence", 
    "Owuo Atwedee": "mortality", 
    "Pempamsie": "readiness, steadfastness", 
    "Sankofa": "learn from the past", 
    "Sesa wo suban": "transformation",
    "Tamfo bebre": "jealousy",
    "Wawa aba": "hardiness, toughness, perseverance",
    "Wo nsa da mu a": "democracy, pluralism",
    "Woforo dua pa a": "support, cooperation",
    "kwatakye atiko": "bravery, valor",}

MEANING = {x:MEANING[x] for x in trainLoader.dataset.classes}

#print(len(MEANING))
