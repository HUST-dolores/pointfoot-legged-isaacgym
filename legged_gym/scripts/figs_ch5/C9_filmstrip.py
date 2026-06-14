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
def Rs(p,b): c,s=np.cos(b),np.sin(b); return np.array([p[0]*c-p[1]*s,p[0]*s+p[1]*c])
def fk(X,Z,th,qh,qk):
    b=np.array([X,Z]);h=b+roty(P_HIP,th);k=h+roty(THIGH,th+qh);w=k+roty(SHANK,th+qh-qk);return b,h,k,w
def o(a): return (a if a.shape[0]>=a.shape[1] else a.T).astype(float)
def loadf(tag):
    f=max(glob.glob(os.path.join(ED,'play_data_2026*%s_walk_vx0.5_seed3_ckpt11000_load23-23_flat_slope20.mat'%tag)),key=os.path.getmtime); S=sio.loadmat(f)
    return dict(pit=o(S['base_pitch_all'])[:,0],px=o(S['base_pos_x_all'])[:,0],hz=o(S['base_height_all'])[:,0],
                qh=o(S['joint_pos_hip_L'])[:,0],qk=o(S['joint_pos_knee_L'])[:,0])
def robot(ax,d,i,beta,refx,col,alpha):
    sh=lambda p:Rs(np.array(p)-np.array([refx,0]),beta)
    th=-d['pit'][i]; b,h,k,w=fk(d['px'][i],d['hz'][i],th,d['qh'][i],d['qk'][i])
    dz=-w[1]; b=b+[0,dz]; h=h+[0,dz]; k=k+[0,dz]; w=w+[0,dz]   # 轮子贴地(消除生成/坠落时的高度漂移)
    B,H,K,W=sh(b),sh(h),sh(k),sh(w)
    z=4 if alpha<1 else 6
    ax.plot([H[0],K[0]],[H[1],K[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([K[0],W[0]],[K[1],W[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([B[0],H[0]],[B[1],H[1]],"-",c="#222",lw=2,zorder=z,alpha=alpha); ax.add_patch(plt.Circle(W,0.095,fc="#777",ec="k",zorder=z,alpha=alpha))
    bx=np.array([[-0.16,-0.05],[0.16,-0.05],[0.16,0.11],[-0.16,0.11]])
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in bx],closed=True,fc=col,ec="k",alpha=alpha*0.92,zorder=z))
    lx=np.array([[-0.06,0.11],[0.06,0.11],[0.06,0.23],[-0.06,0.23]])
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in lx],closed=True,fc="#e0902a",ec="k",alpha=alpha,zorder=z))
    return np.array(sh(b)),np.array(W)
def incline(ax,beta,refx,xr):
    gx=np.linspace(xr[0]-1,xr[1]+2,80); pts=np.array([Rs(np.array([x+refx,0])-np.array([refx,0]),beta) for x in gx])
    ax.fill_between(pts[:,0],pts[:,1]-3,pts[:,1],color="#e8dcc8",zorder=0); ax.plot(pts[:,0],pts[:,1],"-",c="#8a6d3b",lw=1.6,zorder=1)
M=loadf('qs1_resid1'); R=loadf('qs0_resid0_torq0'); beta=np.radians(20); cM,cR="#2c6fbb","#d8551a"
fig,axs=plt.subplots(2,1,figsize=(9.5,7.0))
# --- Model: position extremes start(i=250)->farthest(i=999) ---
ax=axs[0]; refx=M['px'][250]; incline(ax,beta,refx,(-1,10))
BA,WA=robot(ax,M,250,beta,refx,cM,0.32); BB,WB=robot(ax,M,999,beta,refx,cM,1.0)
ax.add_patch(FancyArrowPatch((WA[0],WA[1]-0.2),(WB[0],WB[1]-0.2),arrowstyle="-|>",mutation_scale=22,color="#1f7a1f",lw=2.5,zorder=9))
ax.text((WA[0]+WB[0])/2,(WA[1]+WB[1])/2-0.55,"持续上行 +%.1f m"%(M['px'][999]-M['px'][250]),fontproperties=cjk,fontsize=11,ha="center",color="#1f7a1f",fontweight="bold")
ax.text(-0.04,0.5,"Model-guided\n稳住→爬升",transform=ax.transAxes,fontproperties=cjk,fontsize=13,color=cM,fontweight="bold",ha="right",va="center")
ax.set_xlim(-0.8,9.6); ax.set_ylim(-0.5,4.2); ax.set_aspect("equal"); ax.axis("off")
# --- RL: pitch extremes 最仰(i=1)->最俯/toppled(i=40) + rotation arrow ---
ax=axs[1]; refx=R['px'][0]; incline(ax,beta,refx,(-1,2))
BA,WA=robot(ax,R,0,beta,refx,cR,0.32); BB,WB=robot(ax,R,50,beta,refx,cR,1.0)
ax.add_patch(FancyArrowPatch((BA[0]-0.15,BA[1]+0.55),(BB[0]+0.15,BB[1]+0.35),connectionstyle="arc3,rad=-0.75",arrowstyle="-|>",mutation_scale=26,color="#c0392b",lw=3,zorder=10))
ax.text(BA[0]+0.35,BA[1]+0.95,"前倾倾覆",fontproperties=cjk,fontsize=12,ha="center",color="#c0392b",fontweight="bold")
ax.text(-0.04,0.5,"RL-only\n失稳→倾覆",transform=ax.transAxes,fontproperties=cjk,fontsize=13,color=cR,fontweight="bold",ha="right",va="center")
ax.set_xlim(-1.0,2.0); ax.set_ylim(-0.5,1.5); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("同一 23 kg 负载、20° 坡:Model-guided 持续上行 vs RL-only 前倾倾覆(淡=初态,实=末态)",fontproperties=cjk,fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0.1,0,1,0.95]); plt.savefig(r"legged_gym/scripts/figs_ch5/paper_pdf/C9_起步过程对比.pdf"); plt.savefig("/tmp/_chk.png",dpi=130)
print("Model climb=%.1f  RL pitch %d->%d"%(M['px'][999]-M['px'][250],R['pit'][1]*180/np.pi,R['pit'][40]*180/np.pi))
