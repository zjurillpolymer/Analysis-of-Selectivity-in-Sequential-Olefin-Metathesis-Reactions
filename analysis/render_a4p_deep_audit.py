#!/usr/bin/env python3
"""Render reproducible diagnostic panels and a Chinese evidence report."""
import json
import math
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/a4p-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_a4p_audit import OUT, profile
from analyze_a4p_downloads import RT

def main():
    data=json.loads((OUT/'audit.json').read_text())
    pilot=json.loads((OUT/'pilot_reference_audit.json').read_text())
    comparisons=data['comparisons'];rows=data['pathways']
    labels=['C6 q1 cross/same','C6 q2 adj. switch/same','C6 q2 opp. switch/reuse',
            'C8 q1 cross/same','C8 q2 adj. switch/same']
    colors=['#9B562E','#D09361','#6C655E','#285F8F','#79A8BE']
    markers=['o','s','^','D','v']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'legend.fontsize':6,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,155/25.4),layout='constrained')
    ax=axes[0,0]
    for i,c in enumerate(comparisons):
        for key,offset,col,m in [('Kratio',-.14,'#355E7A','o'),('Pratio_truncated',.14,'#B37742','s')]:
            v=c['time']['full'][key];lo,_,hi=c['bootstrap_conditional_95pct'][key]
            ax.errorbar(v,i+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],color=col,fmt=m,markersize=3,capsize=2,lw=.75)
    ax.axvline(1,color='.6',ls=':',lw=.8)
    ax.set_xscale('log');ax.set_yticks(range(5),labels);ax.invert_yaxis()
    ax.set_xlabel('Selected-pair ratio (conditional 95% bootstrap)')
    ax.set_title('Normalization changes the numerical comparison',loc='left')
    ax.plot([],[],color='#355E7A',marker='o',ls='',label='Contact/reference ratio')
    ax.plot([],[],color='#B37742',marker='s',ls='',label='Contact probability ratio in sampled domain')
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.34),fontsize=5.8)
    ax=axes[0,1]
    for c,label,col,m in zip(comparisons,labels,colors,markers):
        ax.plot([150,250,350,450],[c['time'][b]['Kratio'] for b in ['b1','b2','b3','b4']],color=col,marker=m,ms=3,lw=1,label=label)
    ax.axhline(1,color='.6',ls=':',lw=.8);ax.set_yscale('log')
    ax.set_xlabel('Midpoint of a 100 ps production block (ps)');ax.set_ylabel('Contact/reference ratio')
    ax.set_title('C6 q1 has substantial time dependence',loc='left')
    ax.legend(loc='upper right',fontsize=5.2)
    ax=axes[1,0]
    for c,col,m in zip(comparisons,colors,markers):
        ax.plot(range(5),[x['Kratio'] for x in c['reference_sensitivity']],color=col,marker=m,ms=3,lw=1)
    ax.axhline(1,color='.6',ls=':',lw=.8);ax.set_yscale('log')
    ax.set_xticks(range(5),['0.60–1.00','0.75–1.25','0.90–1.50','1.20–1.80','1.50–2.10'],rotation=25,ha='right')
    ax.set_xlabel('Reference distance interval (nm)');ax.set_ylabel('Contact/reference ratio')
    ax.set_title('Reference interval changes the ratio',loc='left')
    ax=axes[1,1]
    for system,col in [('C6','#9B562E'),('C8','#285F8F')]:
        r=next(r for r in rows if r['metadata']['state']==f'A4_p_{system}_1_cis' and r['metadata']['topology_class']=='cross')
        d=profile(OUT/r['pathway'],'full');finite=np.isfinite(d[:,1]);p=np.zeros(len(d))
        p[finite]=np.exp(-(d[finite,1]-np.min(d[finite,1]))/RT);p/=np.trapz(p,d[:,0])
        ax.plot(d[:,0],p,color=col,lw=1,label=f"{system}: {100*r['tail_mass_above_2_5']:.1f}% above 2.5 nm")
    ax.axvspan(2.5,3,color='.92',zorder=-1);ax.set_xlim(.3,3)
    ax.set_xlabel('Selected vinyl-pair distance (nm)');ax.set_ylabel('Finite-domain density (nm$^{-1}$)')
    ax.set_title('A large distant tail remains for C6 q1 cross',loc='left');ax.legend(loc='upper left')
    for tag,ax in zip('abcd',axes.flat):
        ax.text(-.12,1.06,tag,transform=ax.transAxes,fontweight='bold',fontsize=9,va='top')
    fig.savefig(OUT/'diagnostics.svg')
    fig.savefig(OUT/'diagnostics.png',dpi=300)
    plt.close(fig)
    lines=['# A4-p-C6/C8：深入核查与结论修订','',
           '## 结论先行','',
           '本轮275个窗口在计算层面均已完成；恢复窗口可用。但当前证据只支持所选乙烯基对的局部构象趋势，尚不能确认可转移的单一α。',
           '优先级最高的问题不是再增加拟合项，而是修正A4邻接规则的实现、明确概率的归一化定义，并检验时间与独立初态收敛。',
           '此前“C8缺少reuse”的表述应更正为“编号邻接过滤排除了C8的reuse候选，相关PMF没有被提交”。', '',
           '## 1. 四臂编号不是环形骨架次序','',
           '按未反应态PDB的重原子连接图，两个A4分子均可在中心碳（序号1）处分成四个分支；每个分支恰含两个末端乙烯基。C6每支34个重原子，C8每支38个。',
           '历史脚本的A1–A4来自“同臂乙烯基对按编号排序”，而不是沿环形骨架顺次编号。随后将A1–A2、A2–A3、A3–A4、A4–A1硬编码为邻居，缺少独立结构依据。',
           '这不否定用户对B6相邻位置的模型假设；它说明不能把该假设以任意臂编号的形式直接套到当前四臂中心骨架。若A4也需施加邻居约束，必须明确其结构/几何邻接定义。','',
           '| C8 opposite候选 | 臂对 | 与反应历史的关系 | 旧编号规则 |','|---|---|---|---|',
           '| V1–V5 | A1–A3 | 复用q1边，reuse | 被排除 |',
           '| V3–V6 | A2–A4 | 复用q2边，reuse | 被排除 |',
           '| V1–V3、V1–V6、V3–V5、V5–V6 | 另外四个臂对 | switch | 保留 |','',
           'q1在旧规则中被统一标为cross而未拆成reuse/switch；已做的两条q1 cross代表都是switch候选，因此也不能用q1 cross/same替代q1 reuse/switch。',
           'adjacent/opposite分支名称描述第二条反应边与第一条边是否共享臂，不应混同为四个臂在空间中的邻位/对位。','',
           '## 2. 先前比值的物理含义必须收紧','',
           r'设接触区C=[0.35,0.50] nm，参考区R=[0.90,1.50] nm。原指标是 $K_i=P_i(C)/P_i(R)$。',
           r'因此 $K_a/K_b=[P_a(C)/P_b(C)]\,[P_b(R)/P_a(R)]$；除非两条路径参考区占比相同，否则它不等于接触概率比。',
           '另外计算了同一距离范围D=[0.15,3.00] nm内归一化的接触概率比，用于检查归一化敏感性。这仍是有限范围条件概率，不是已证明的完整平衡系综概率。','',
           '| 所选路径对 | 原K比 | D内接触概率比 | 原K比条件bootstrap 95% |','|---|---:|---:|---|']
    for c,label in zip(comparisons,labels):
        q=c['bootstrap_conditional_95pct']['Kratio']
        lines.append(f"| {label} | {c['time']['full']['Kratio']:.3g} | {c['time']['full']['Pratio_truncated']:.3g} | {q[0]:.3g}–{q[2]:.3g} |")
    lines += ['', '尤其是q1：原K比给出C6≈125、C8≈15.6，而D内概率比给出C6≈16.9、C8≈76.2。不能挑选其中一种归一化来宣称臂长趋势；先要确定目标观测量。',
              '参考区间移动后，C8 q1的K比从约2.79变到152；C6 opposite的K比可从0.272变到1.013。它们是不同参考态定义下的不同统计量。','',
              '## 3. 时间收敛与不确定性','',
              '生产段100–500 ps拆成四个连续100 ps区间；所有窗口都采用相同时间边界。',
              '| 比较 | 100–200 ps | 200–300 ps | 300–400 ps | 400–500 ps |','|---|---:|---:|---:|---:|']
    for c,label in zip(comparisons,labels):
        lines.append('| '+label+' | '+' | '.join(f"{c['time'][b]['Kratio']:.3g}" for b in ['b1','b2','b3','b4'])+' |')
    opp=next(c for c in comparisons if c['reference']=='reuse')
    ci=opp['bootstrap_conditional_95pct']['Pratio_truncated']
    alpha=math.log(opp['time']['full']['Pratio_truncated'])
    lines += ['', 'C6 q1原K比从约1445降到10；C6 adjacent也存在明显漂移。窗口重叠良好不能证明其他构象自由度已经平衡。',
              'C6 opposite的K比95%区间跨1；D内接触概率比在第一个时间块偏switch，后三块偏reuse，当前只能说中心估计偏reuse，不能确认稳定偏好。',
              f'若仅作诊断，按此前约定α=ln(Pswitch/Preuse)，C6 opposite的有限范围单对估计为{alpha:.3f}，条件bootstrap区间约[{math.log(ci[0]):.3f}, {math.log(ci[2]):.3f}]，跨0。原参考区指标对应ln(Kswitch/Kreuse)≈−0.659；二者不能混用为同一个已标定α。',
              '每窗口只有1条轨迹。100条bootstrap是由已采样直方图生成的合成轨迹，不是100次独立MD；两条路径的bootstrap经独立配对得到比值区间。这里的不确定度尚未包含隐藏构象盆地与不同初态间差异。', '',
              '## 4. 与三重复无偏pilot核对','',
              '重新直接读取各状态3条2 ns轨迹，在200–2000 ps统计同样的距离区间，未施加编号环过滤。仅作为短pilot诊断，不据此替代umbrella。',
              '| 所选路径 | 三重复接触占比（%） | 三重复参考区占比（%） |','|---|---|---|']
    for r in rows:
        m=r['metadata'];pair=f"V{m['left']}-V{m['right']}"
        matched=[p for p in pilot['pairs'] if p['state']==m['state'] and p['pair']==pair]
        lines.append(f"| {m['state']} {pair} | "+' / '.join(f"{100*p['Pcontact_035_050']:.3g}" for p in matched)+' | '+' / '.join(f"{100*p['Preference_090_150']:.3g}" for p in matched)+' |')
    lines += ['', 'C6 q1 cross的三个重复接触占比从约0.028%到12.05%，重复不一致；C8 adjacent switch在三个短pilot中均没有进入此接触区。它们提示全局构象采样仍需检查，不宜直接把参考区占比作为精确归一化校正。',
              '被过滤的C8 opposite reuse候选V3–V6，在pilot第1重复实际进入了0.35–0.50 nm（约0.194%，最小距离0.398 nm）。这不是化学反应证据，但足以否定“没做该PMF等于物理上不可接触”的解释。','',
              '## 5. 末窗口修复和边界检查','',
              '重新独立运行25窗口WHAM，并记录TPR/pullx输入SHA-256，避免仅按命令文本复用缓存。修复末窗口的TPR确为k=2000，其他为500；WHAM读取各自实际偏置，没有把所有窗口当成同一个k。',
              '两个恢复窗口与前一窗口的直方图重叠约0.263、0.345；没有检出恢复日志中的PBC致命错误或LINCS警告。修复解决了运行失败，但不会自动解决全局采样不足。',
              'C6 q1 cross在D内归一化后，约52.9%的权重位于2.5 nm以上，约10.9%位于2.8 nm以上。C8 adjacent switch的密度峰也靠近上边界。因而归一化域上端尚未被充分论证；原区间K对末窗口不敏感，不能替代这项检查。','',
              '## 6. 可保留的结论与下一步优先级','',
              '1. 在已选单对和所测时间段内，q1 cross/same以及adjacent switch/same的方向多数一致：所选跨臂路径更容易达到所定义的接触区。数值大小仍需收敛验证，不能冒充全类别结果。',
              '2. C6 opposite最多是偏reuse的初步迹象，α不显著；C8没有有效的reuse PMF比较，不能确认其α符号，更不能沿用A4-m-C5/C8的结论作为A4-p数据的证明。',
              '3. 首先修正A4的邻接定义及历史边一致性检查，再处理C8遗漏的V1–V5/V3–V6；不要继续让任意编号决定候选是否存在。',
              '4. 固定α的概率定义。若使用接触概率，需验证归一化域充分，或用充分收敛的无偏参考区概率校准K；这不是增加经验参数。',
              '5. 之后优先对C6 q1与opposite两类比较、以及pilot/umbrella差异大的C8 adjacent switch增加独立初态和采样；延长同一条轨迹与独立重复不能互相替代。',
              '本轮只做本地重分析，没有新建付费MD任务，没有修改原化学结构或原始归档。','',
              '## 方法与诊断图','',
              'GROMACS WHAM说明与bootstrap限制：https://manual.gromacs.org/documentation/2026.1/onlinehelp/gmx-wham.html',
              '图a是条件bootstrap区间，b是不同时间块而非置信区间，c仅改变参考区间，d是有限域内的归一化距离密度。所有点均来自本地计算，未平滑数据。','']
    (OUT/'DETAILED_REPORT.md').write_text('\n'.join(lines))
    print(OUT/'DETAILED_REPORT.md')

if __name__=='__main__':main()
