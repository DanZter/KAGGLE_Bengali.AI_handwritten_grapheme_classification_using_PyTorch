import os
import ast
import torch
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
from tqdm import tqdm

# PARAMETERS TO TRAIN THE MODEL
DEVICE = "cpu"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT =int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH =int(os.environ.get("IMG_WIDTH"))
EPOCHS =int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE =int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE =int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN =ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD =ast.literal_eval(os.environ.get("MODEL_STD"))
""" not int, we use literal_eval:
because environment variables are string and this is going to be string of tuples or lists.
this converts stringified list to normal list """

TRAINING_FOLDS =ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS =ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL =os.environ.get("BASE_MODEL")

# LOSS FUNCTION

def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1+l2+l3) / 3               # can return weighted averaging of these losses



# TRAINING AND VALIDATION LOOPS

def train(dataset, data_loader, model, optimizer):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter = counter + 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.float)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.float)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.float)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)
        final_loss += loss
    return final_loss/counter
    

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)   # pretrained true cos its training loop
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    """ here we use all parameters.
    we can experiment with different paramaeters, se can also set 
    differential learning rates for different layers"""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                          patience=5, factor=0.3, verbose=True)
    """ some schedulers we need to step every batch or every epoch.
    LR scheduler on plateau: when model scores plateau the scheduler reduces learning rates 
    mode = "min" :if we want to reduce the loss function
    min = "max" : here we will see recall score"""

    if torch.cuda.device_count() >1:   # if u have multiple gpus in system
        model = nn.DataParallel(model)

    """ we can impelement early stopping using:
     https://github.com/Bjarten/early-stopping-pytorch """

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")

if __name__=="__main__":
    main()



