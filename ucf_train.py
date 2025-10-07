import logging
from datetime import datetime
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model import VADModel
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option



# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('training.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def setup_seed(seed):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Set seed to {seed}")


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    logger.info(f"Starting training on {device}")

    # 载入数据
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 初始化模型和优化器
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    # 载入预训练模型（如果需要）
    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_ap = checkpoint['ap']
        logger.info(f"Loaded checkpoint: epoch {start_epoch + 1}, AP {best_ap}")
    else:
        start_epoch = 0
        best_ap = 0

    # 获取提示文本
    prompt_text = get_prompt_text(label_map)

    # 训练循环
    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        # 确保两个数据加载器的长度相同，以便同步迭代
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            # 解包每个批次中的特征和标签
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            # 将正常和异常数据分别处理，而不是拼接在一起
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # 前向传播
            # text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)
            text_features, logits1, logits2 = model(visual_features, prompt_text)


            # 计算损失
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)

            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            # loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)  # normal的维度为0
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            total_loss = loss1 + loss2 + loss3

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2

            if step % 128 == 0 and step != 0:
                logger.info(
                    f'epoch: {e + 1} | step: {step} | loss1: {loss_total1 / (i + 1)} | loss2: {loss_total2 / (i + 1)} | loss3: {loss3.item()}')
                
            if step % (128 * 50) == 0 and step != 0:
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                logger.info(f"Epoch {e + 1}: Val AUC {AUC:.4f}, AP {AP:.4f}")
                if AP > best_ap:
                    best_ap = AP

                    # 保存模型
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': best_ap,}
                    torch.save(checkpoint, args.checkpoint_path)
                    logger.info(f"New best AP {best_ap} saved at {args.checkpoint_path}")

        scheduler.step()
        # torch.save(model.state_dict(), f'model/epoch_{e + 1}.pth')
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 最终保存模型
    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    logger.info("Training completed successfully")


if __name__ == '__main__':

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda:1"
    args = ucf_option.parser.parse_args()
    # setup_seed(args.seed)

    label_map = {
        'Normal': 'normal',
        'Abuse': 'abuse',
        'Arrest': 'arrest',
        'Arson': 'arson',
        'Assault': 'assault',
        'Burglary': 'burglary',
        'Explosion': 'explosion',
        'Fighting': 'fighting',
        'RoadAccidents': 'roadAccidents',
        'Robbery': 'robbery',
        'Shooting': 'shooting',
        'Shoplifting': 'shoplifting',
        'Stealing': 'stealing',
        'Vandalism': 'vandalism'
    }

    # 数据集初始化

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VADModel(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device
    )

    setup_seed(args.seed)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)