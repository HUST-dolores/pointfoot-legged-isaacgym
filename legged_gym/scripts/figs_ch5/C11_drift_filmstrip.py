import os
_HERE=os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ED=os.path.join(_HERE,"..","..","..","logs","wheelfoot_flat","WF_TRON1A","exported")
import os, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch
import scipy.io as sio
cjk=fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"); plt.rcParams["axes.unicode_minus"]=False
P_HIP=np.array([-0.0214,-0.2602]); THIGH=np.array([-0.15,-0.25981]); SHANK=np.array([0.15,-0.25981])
def roty(v,a): c,s=np.cos(a),np.sin(a); return np.array([v[0]*c+v[1]*s,-v[0]*s+v[1]*c])
def fk(X,Z,th,qh,qk):
    b=np.array([X,Z]);h=b+roty(P_HIP,th);k=h+roty(THIGH,th+qh);w=k+roty(SHANK,th+qh-qk);return b,h,k,w
def o(a): return (a if a.shape[0]>=a.shape[1] else a.T).astype(float)
def load(tag):
    f=max(glob.glob(os.path.join(ED,'play_data_2026*%s*_load13-13_flat_ecc0.15x0y.mat'%tag)),key=os.path.getmtime); S=sio.loadmat(f)
    return dict(pit=o(S['base_pitch_all'])[:,0],px=o(S['base_pos_x_all'])[:,0],hz=o(S['base_height_all'])[:,0],
                qh=o(S['joint_pos_hip_L'])[:,0],qk=o(S['joint_pos_knee_L'])[:,0])
def robot(ax,d,i,refx,col,alpha):
    sh=lambda p:np.array([p[0]-refx,p[1]])
    th=-d['pit'][i]; b,h,k,w=fk(d['px'][i],d['hz'][i],th,d['qh'][i],d['qk'][i]); B,H,K,W=sh(b),sh(h),sh(k),sh(w)
    z=4 if alpha<1 else 6
    ax.plot([H[0],K[0]],[H[1],K[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([K[0],W[0]],[K[1],W[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([B[0],H[0]],[B[1],H[1]],"-",c="#222",lw=2,zorder=z,alpha=alpha); ax.add_patch(plt.Circle(W,0.095,fc="#777",ec="k",zorder=z,alpha=alpha))
    bx=np.array([[-0.16,-0.05],[0.16,-0.05],[0.16,0.11],[-0.16,0.11]])
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in bx],closed=True,fc=col,ec="k",alpha=alpha*0.92,zorder=z))
    lx=np.array([[0.07,0.11],[0.21,0.11],[0.21,0.25],[0.07,0.25]])  # 偏心前置负载
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in lx],closed=True,fc="#e0902a",ec="k",alpha=alpha,zorder=z))
    ax.annotate("",xy=sh(b+roty([0.14,0.05],th)),xytext=sh(b+roty([0.14,0.32],th)),arrowprops=dict(arrowstyle="->",color="#b06010",lw=1.2,alpha=alpha),zorder=z+1)
    return W
def panel(ax,d,iA,iB,refx,col,xlim,ylim):
    ax.fill_between([-2,7],[-3,-3],[0,0],color="#e8dcc8",zorder=0); ax.plot([-2,7],[0,0],"-",c="#8a6d3b",lw=1.6,zorder=1)
    ax.plot([0,0],[0,0.6],"-",c="#444",lw=1.2,zorder=2)
    ax.add_patch(plt.Polygon([[0,0.6],[0.28,0.5],[0,0.4]],closed=True,fc="#e03b3b",ec="k",zorder=2)); ax.text(-0.05,0.62,"起点",fontproperties=cjk,fontsize=9,ha="center",va="bottom",color="#444")
    WA=robot(ax,d,iA,refx,col,0.32); WB=robot(ax,d,iB,refx,col,1.0)
    ax.add_patch(FancyArrowPatch((WA[0],WA[1]-0.18),(WB[0],WB[1]-0.18),arrowstyle="-|>",mutation_scale=22,color="#c0392b",lw=2.5,zorder=9))
    ax.text((WA[0]+WB[0])/2,-0.42,"漂移 %+.2f m"%(d['px'][iB]-d['px'][iA]),fontproperties=cjk,fontsize=11,ha="center",color="#c0392b",fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
M=load('qs1_resid1'); R=load('qs0_resid0_torq0'); cM,cR="#2c6fbb","#d8551a"; i0=25
# start (faded) -> 末态最大位移点 (solid)
iAM,iBM=i0,i0+int(np.argmax(np.abs(M['px'][i0:]-M['px'][i0]))); iAR,iBR=i0,i0+int(np.argmax(np.abs(R['px'][i0:]-R['px'][i0])))
xlim,ylim=(-1.0,5.2),(-0.55,1.0)
fig,axs=plt.subplots(2,1,figsize=(9.2,5.6))
panel(axs[0],M,iAM,iBM,M['px'][i0],cM,xlim,ylim); axs[0].text(-0.04,0.5,"Model-guided\n守住原位",transform=axs[0].transAxes,fontproperties=cjk,fontsize=13,color=cM,fontweight="bold",ha="right",va="center")
panel(axs[1],R,iAR,iBR,R['px'][i0],cR,xlim,ylim); axs[1].text(-0.04,0.5,"RL-only\n被推着漂移",transform=axs[1].transAxes,fontproperties=cjk,fontsize=13,color=cR,fontweight="bold",ha="right",va="center")
fig.suptitle("搭载 13 kg 前置偏心负载、平地静止:起点→最大位移(淡=起点,实=最远)",fontproperties=cjk,fontsize=13.5,fontweight="bold")
plt.tight_layout(rect=[0.1,0,1,0.94]); plt.savefig(r"legged_gym/scripts/figs_ch5/paper_pdf/C11_偏心定点漂移帧条.pdf"); plt.savefig("/tmp/_chk.png",dpi=130)
print("Model drift=%.2f RL drift=%.2f"%(M['px'][iBM]-M['px'][iAM],R['px'][iBR]-R['px'][iAR]))
