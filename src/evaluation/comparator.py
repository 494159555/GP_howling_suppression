'''
方法对比工具模块

提供多种啸叫抑制方法的对比分析功能，包括：
- 方法性能对比
- 统计显著性检验
- 排名和推荐
- 详细对比报告
'''

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
from pathlib import Path
from .metrics import AudioMetrics, calculate_mos_score


class MethodComparator:
    """方法对比分析类"""
    
    def __init__(self, metrics_calculator: AudioMetrics = None):
        self.metrics_calculator = metrics_calculator or AudioMetrics()
        self.comparison_results = {}
        
    def compare_methods(self, results_dict: Dict[str, Dict[str, float]],
                      significance_level: float = 0.05) -> Dict:
        """
        对比多种方法的性能
        
        Args:
            results_dict: 方法结果字典 {method_name: {metric: value}}
            significance_level: 显著性水平
            
        Returns:
            对比结果字典
        """
        methods = list(results_dict.keys())
        
        # 1. 基本统计信息
        basic_stats = self._calculate_basic_stats(results_dict)
        
        # 2. 方法排名
        rankings = self._calculate_rankings(results_dict)
        
        # 3. 统计显著性检验 (如果有多个样本)
        significance_tests = self._perform_significance_tests(
            results_dict, significance_level
        )
        
        # 4. 综合评分和推荐
        comprehensive_scores = self._calculate_comprehensive_scores(results_dict)
        recommendations = self._generate_recommendations(comprehensive_scores, results_dict)
        
        # 5. 优缺点分析
        analysis = self._analyze_strengths_weaknesses(results_dict)
        
        comparison_results = {
            'methods': methods,
            'basic_stats': basic_stats,
            'rankings': rankings,
            'significance_tests': significance_tests,
            'comprehensive_scores': comprehensive_scores,
            'recommendations': recommendations,
            'analysis': analysis,
            'summary': self._generate_summary(comprehensive_scores, rankings)
        }
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _calculate_basic_stats(self, results_dict: Dict[str, Dict[str, float]]) -> Dict:
        """计算基本统计信息"""
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db']
        
        stats = {}
        for metric in metrics:
            values = [results_dict[method].get(metric, 0) for method in methods]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
        
        return stats
    
    def _calculate_rankings(self, results_dict: Dict[str, Dict[str, float]]) -> Dict:
        """计算各指标的方法排名"""
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db',
                  'processing_time_ms', 'memory_usage_mb', 'parameter_count']
        
        rankings = {}
        for metric in metrics:
            values = []
            for method in methods:
                value = results_dict[method].get(metric, 0)
                values.append(value)
            
            # 对于时间、内存、参数量，越小越好
            if metric in ['processing_time_ms', 'memory_usage_mb', 'parameter_count']:
                ranks = stats.rankdata([-v for v in values])  # 反向排名
            else:
                ranks = stats.rankdata([-v for v in values])  # 越大越好
            
            rankings[metric] = dict(zip(methods, ranks))
        
        return rankings
    
    def _perform_significance_tests(self, results_dict: Dict[str, Dict[str, float]],
                                 significance_level: float) -> Dict:
        """执行统计显著性检验"""
        # 注意：这里简化处理，实际需要多个样本的数据
        # 实际应用中应该收集多次运行的结果
        
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db']
        
        significance_results = {}
        
        for metric in metrics:
            significance_results[metric] = {}
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    value1 = results_dict[method1].get(metric, 0)
                    value2 = results_dict[method2].get(metric, 0)
                    
                    # 简化的t检验（实际需要更多样本）
                    # 这里使用简单的差异分析
                    diff = abs(value1 - value2)
                    threshold = 0.1 * max(abs(value1), abs(value2))  # 简化阈值
                    
                    is_significant = diff > threshold
                    
                    significance_results[metric][f"{method1}_vs_{method2}"] = {
                        'difference': diff,
                        'threshold': threshold,
                        'is_significant': is_significant,
                        'better_method': method1 if value1 > value2 else method2
                    }
        
        return significance_results
    
    def _calculate_comprehensive_scores(self, results_dict: Dict[str, Dict[str, float]]) -> Dict:
        """计算综合评分"""
        methods = list(results_dict.keys())
        
        # 定义权重
        weights = {
            'snr_improvement_db': 0.25,
            'psnr_db': 0.20,
            'stoi_score': 0.25,
            'howling_reduction_db': 0.20,
            'processing_time_ms': -0.05,  # 负权重，时间越短越好
            'memory_usage_mb': -0.03,     # 负权重，内存越少越好
            'parameter_count': -0.02       # 负权重，参数越少越好
        }
        
        comprehensive_scores = {}
        
        for method in methods:
            score = 0.0
            
            # 质量指标评分
            for metric, weight in weights.items():
                if weight > 0:  # 正向指标
                    value = results_dict[method].get(metric, 0)
                    # 归一化
                    if metric == 'snr_improvement_db':
                        norm_value = min(1.0, max(0.0, value / 20))
                    elif metric == 'psnr_db':
                        norm_value = min(1.0, max(0.0, value / 40))
                    elif metric == 'stoi_score':
                        norm_value = min(1.0, max(0.0, value))
                    elif metric == 'howling_reduction_db':
                        norm_value = min(1.0, max(0.0, value / 10))
                    else:
                        norm_value = 0
                    
                    score += weight * norm_value
                
                else:  # 负向指标（越小越好）
                    value = results_dict[method].get(metric, 1)
                    # 归一化并反向
                    if metric == 'processing_time_ms':
                        norm_value = max(0.0, 1.0 - min(1.0, value / 1000))
                    elif metric == 'memory_usage_mb':
                        norm_value = max(0.0, 1.0 - min(1.0, value / 1000))
                    elif metric == 'parameter_count':
                        norm_value = max(0.0, 1.0 - min(1.0, value / 1000000))
                    else:
                        norm_value = 1.0
                    
                    score += abs(weight) * norm_value
            
            comprehensive_scores[method] = {
                'score': score,
                'mos_estimate': calculate_mos_score(results_dict[method]),
                'quality_rank': 0,  # 稍后填充
                'efficiency_rank': 0  # 稍后填充
            }
        
        # 计算排名
        quality_scores = {m: s['mos_estimate'] for m, s in comprehensive_scores.items()}
        efficiency_scores = {m: results_dict[m].get('processing_time_ms', 1000) 
                          for m in methods}
        
        quality_ranks = stats.rankdata([-v for v in quality_scores.values()])
        efficiency_ranks = stats.rankdata([v for v in efficiency_scores.values()])
        
        for i, method in enumerate(methods):
            comprehensive_scores[method]['quality_rank'] = quality_ranks[i]
            comprehensive_scores[method]['efficiency_rank'] = efficiency_ranks[i]
        
        return comprehensive_scores
    
    def _generate_recommendations(self, comprehensive_scores: Dict,
                                results_dict: Dict) -> Dict:
        """生成推荐建议"""
        methods = list(comprehensive_scores.keys())
        
        # 按综合评分排序
        sorted_methods = sorted(methods, 
                              key=lambda x: comprehensive_scores[x]['score'], 
                              reverse=True)
        
        recommendations = {
            'best_overall': sorted_methods[0],
            'best_quality': max(methods, key=lambda x: comprehensive_scores[x]['mos_estimate']),
            'most_efficient': min(methods, key=lambda x: results_dict[x].get('processing_time_ms', float('inf'))),
            'lightest': min(methods, key=lambda x: results_dict[x].get('memory_usage_mb', float('inf'))),
            'best_for_realtime': None,  # 稍后计算
            'best_for_high_quality': None,  # 稍后计算
            'detailed_recommendations': {}
        }
        
        # 实时应用推荐（处理时间 < 100ms）
        realtime_candidates = [m for m in methods 
                             if results_dict[m].get('processing_time_ms', float('inf')) < 100]
        if realtime_candidates:
            recommendations['best_for_realtime'] = max(realtime_candidates,
                                                      key=lambda x: comprehensive_scores[x]['score'])
        
        # 高质量应用推荐（MOS > 4.0）
        quality_candidates = [m for m in methods 
                            if comprehensive_scores[m]['mos_estimate'] > 4.0]
        if quality_candidates:
            recommendations['best_for_high_quality'] = max(quality_candidates,
                                                          key=lambda x: comprehensive_scores[x]['score'])
        
        # 详细推荐
        for method in methods:
            score_info = comprehensive_scores[method]
            metrics_info = results_dict[method]
            
            if method == recommendations['best_overall']:
                rec_type = "综合最佳"
            elif method == recommendations['best_quality']:
                rec_type = "质量最佳"
            elif method == recommendations['most_efficient']:
                rec_type = "效率最高"
            elif method == recommendations['best_for_realtime']:
                rec_type = "实时应用推荐"
            elif method == recommendations['best_for_high_quality']:
                rec_type = "高质量应用推荐"
            else:
                rec_type = "特定场景适用"
            
            recommendations['detailed_recommendations'][method] = {
                'recommendation_type': rec_type,
                'strengths': self._identify_strengths(method, results_dict, comprehensive_scores),
                'weaknesses': self._identify_weaknesses(method, results_dict, comprehensive_scores),
                'best_use_case': self._suggest_use_case(method, results_dict, comprehensive_scores)
            }
        
        return recommendations
    
    def _identify_strengths(self, method: str, results_dict: Dict, 
                          comprehensive_scores: Dict) -> List[str]:
        """识别方法优势"""
        strengths = []
        metrics = results_dict[method]
        
        # 质量优势
        if comprehensive_scores[method]['mos_estimate'] > 4.0:
            strengths.append("音质优秀")
        elif comprehensive_scores[method]['mos_estimate'] > 3.5:
            strengths.append("音质良好")
        
        # 效率优势
        if metrics.get('processing_time_ms', 0) < 50:
            strengths.append("处理速度极快")
        elif metrics.get('processing_time_ms', 0) < 100:
            strengths.append("处理速度快")
        
        # 内存优势
        if metrics.get('memory_usage_mb', 0) < 50:
            strengths.append("内存占用少")
        
        # 啸叫抑制优势
        if metrics.get('howling_reduction_db', 0) > 10:
            strengths.append("啸叫抑制效果显著")
        
        # 参数优势
        if metrics.get('parameter_count', 0) < 10000:
            strengths.append("模型轻量")
        
        return strengths if strengths else ["无明显优势"]
    
    def _identify_weaknesses(self, method: str, results_dict: Dict,
                           comprehensive_scores: Dict) -> List[str]:
        """识别方法劣势"""
        weaknesses = []
        metrics = results_dict[method]
        
        # 质量劣势
        if comprehensive_scores[method]['mos_estimate'] < 3.0:
            weaknesses.append("音质有待改善")
        
        # 效率劣势
        if metrics.get('processing_time_ms', 0) > 500:
            weaknesses.append("处理速度较慢")
        
        # 内存劣势
        if metrics.get('memory_usage_mb', 0) > 500:
            weaknesses.append("内存占用较大")
        
        # 啸叫抑制劣势
        if metrics.get('howling_reduction_db', 0) < 3:
            weaknesses.append("啸叫抑制效果有限")
        
        # 参数劣势
        if metrics.get('parameter_count', 0) > 1000000:
            weaknesses.append("模型参数量大")
        
        return weaknesses if weaknesses else ["无明显劣势"]
    
    def _suggest_use_case(self, method: str, results_dict: Dict,
                         comprehensive_scores: Dict) -> str:
        """建议使用场景"""
        metrics = results_dict[method]
        mos = comprehensive_scores[method]['mos_estimate']
        time_ms = metrics.get('processing_time_ms', 0)
        memory_mb = metrics.get('memory_usage_mb', 0)
        
        # 实时应用
        if time_ms < 50 and memory_mb < 100:
            return "实时通信、直播等对延迟敏感的应用"
        
        # 高质量应用
        elif mos > 4.0:
            return "音乐制作、广播等对音质要求高的应用"
        
        # 移动应用
        elif memory_mb < 50 and metrics.get('parameter_count', 0) < 100000:
            return "移动设备、嵌入式系统等资源受限环境"
        
        # 通用应用
        elif 3.0 < mos < 4.0 and 50 < time_ms < 200:
            return "一般音频处理应用"
        
        else:
            return "特定研究或实验用途"
    
    def _analyze_strengths_weaknesses(self, results_dict: Dict) -> Dict:
        """分析各方法的优缺点"""
        methods = list(results_dict.keys())
        analysis = {}
        
        for method in methods:
            metrics = results_dict[method]
            
            # 相对优势分析
            advantages = []
            disadvantages = []
            
            # 与其他方法比较
            other_methods = [m for m in methods if m != method]
            
            if other_methods:
                # SNR改善
                snr_avg = np.mean([results_dict[m].get('snr_improvement_db', 0) for m in other_methods])
                if metrics.get('snr_improvement_db', 0) > snr_avg * 1.1:
                    advantages.append("SNR改善优于平均水平")
                elif metrics.get('snr_improvement_db', 0) < snr_avg * 0.9:
                    disadvantages.append("SNR改善低于平均水平")
                
                # 处理时间
                time_avg = np.mean([results_dict[m].get('processing_time_ms', 0) for m in other_methods])
                if metrics.get('processing_time_ms', 0) < time_avg * 0.8:
                    advantages.append("处理速度优于平均水平")
                elif metrics.get('processing_time_ms', 0) > time_avg * 1.2:
                    disadvantages.append("处理速度慢于平均水平")
                
                # 内存使用
                memory_avg = np.mean([results_dict[m].get('memory_usage_mb', 0) for m in other_methods])
                if metrics.get('memory_usage_mb', 0) < memory_avg * 0.8:
                    advantages.append("内存使用优于平均水平")
                elif metrics.get('memory_usage_mb', 0) > memory_avg * 1.2:
                    disadvantages.append("内存使用高于平均水平")
            
            analysis[method] = {
                'advantages': advantages if advantages else ["表现均衡"],
                'disadvantages': disadvantages if disadvantages else ["无明显短板"],
                'key_characteristics': self._extract_key_characteristics(metrics)
            }
        
        return analysis
    
    def _extract_key_characteristics(self, metrics: Dict) -> List[str]:
        """提取关键特征"""
        characteristics = []
        
        # 基于指标值判断特征
        if metrics.get('snr_improvement_db', 0) > 15:
            characteristics.append("强噪声抑制")
        elif metrics.get('snr_improvement_db', 0) > 10:
            characteristics.append("中等噪声抑制")
        else:
            characteristics.append("轻度噪声抑制")
        
        if metrics.get('processing_time_ms', 0) < 50:
            characteristics.append("超低延迟")
        elif metrics.get('processing_time_ms', 0) < 200:
            characteristics.append("低延迟")
        else:
            characteristics.append("高延迟")
        
        if metrics.get('parameter_count', 0) > 1000000:
            characteristics.append("大型模型")
        elif metrics.get('parameter_count', 0) > 100000:
            characteristics.append("中型模型")
        else:
            characteristics.append("轻量模型")
        
        return characteristics
    
    def _generate_summary(self, comprehensive_scores: Dict, rankings: Dict) -> Dict:
        """生成对比总结"""
        methods = list(comprehensive_scores.keys())
        
        # 按综合评分排序
        sorted_methods = sorted(methods, 
                              key=lambda x: comprehensive_scores[x]['score'], 
                              reverse=True)
        
        summary = {
            'overall_winner': sorted_methods[0],
            'top_3_methods': sorted_methods[:3],
            'key_findings': [],
            'performance_tiers': self._categorize_performance_tiers(comprehensive_scores)
        }
        
        # 关键发现
        if len(methods) >= 2:
            best_score = comprehensive_scores[sorted_methods[0]]['score']
            worst_score = comprehensive_scores[sorted_methods[-1]]['score']
            
            if best_score > worst_score * 1.5:
                summary['key_findings'].append("方法间性能差异显著")
            else:
                summary['key_findings'].append("方法间性能差异较小")
            
            # 效率vs质量权衡
            quality_leader = max(methods, key=lambda x: comprehensive_scores[x]['mos_estimate'])
            efficiency_leader = min(methods, key=lambda x: comprehensive_scores[x]['score'])
            
            if quality_leader != efficiency_leader:
                summary['key_findings'].append("存在效率与质量的权衡")
            else:
                summary['key_findings'].append("存在兼顾效率与质量的方法")
        
        return summary
    
    def _categorize_performance_tiers(self, comprehensive_scores: Dict) -> Dict:
        """将方法分类到不同性能层级"""
        methods = list(comprehensive_scores.keys())
        scores = [comprehensive_scores[m]['score'] for m in methods]
        
        # 使用分位数分类
        q75 = np.percentile(scores, 75)
        q50 = np.percentile(scores, 50)
        q25 = np.percentile(scores, 25)
        
        tiers = {
            'excellent': [],
            'good': [],
            'average': [],
            'below_average': []
        }
        
        for method in methods:
            score = comprehensive_scores[method]['score']
            if score >= q75:
                tiers['excellent'].append(method)
            elif score >= q50:
                tiers['good'].append(method)
            elif score >= q25:
                tiers['average'].append(method)
            else:
                tiers['below_average'].append(method)
        
        return tiers
    
    def save_comparison_report(self, save_path: str = "comparison_report.json"):
        """保存对比报告"""
        if not self.comparison_results:
            raise ValueError("请先运行 compare_methods 方法")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, ensure_ascii=False, indent=2)
        
        return str(save_path)
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        if not self.comparison_results:
            raise ValueError("请先运行 compare_methods 方法")
        
        methods = self.comparison_results['methods']
        
        # 准备表格数据
        table_data = []
        for method in methods:
            row = {'方法': method}
            
            # 添加各项指标
            if method in self.comparison_results.get('comprehensive_scores', {}):
                score_info = self.comparison_results['comprehensive_scores'][method]
                row.update({
                    '综合评分': f"{score_info['score']:.3f}",
                    'MOS估算': f"{score_info['mos_estimate']:.2f}",
                    '质量排名': score_info['quality_rank'],
                    '效率排名': score_info['efficiency_rank']
                })
            
            # 添加推荐信息
            if method in self.comparison_results.get('recommendations', {}).get('detailed_recommendations', {}):
                rec_info = self.comparison_results['recommendations']['detailed_recommendations'][method]
                row.update({
                    '推荐类型': rec_info['recommendation_type'],
                    '适用场景': rec_info['best_use_case']
                })
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
