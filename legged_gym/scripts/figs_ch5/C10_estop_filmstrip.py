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
def load(tag):
    f=max(glob.glob(os.path.join(ED,'play_data_2026*%s*_load26-26_flat_slope10_estop1.5.mat'%tag)),key=os.path.getmtime); S=sio.loadmat(f)
    return dict(pit=o(S['base_pitch_all'])[:,0],px=o(S['base_pos_x_all'])[:,0],hz=o(S['base_height_all'])[:,0],
                qh=o(S['joint_pos_hip_L'])[:,0],qk=o(S['joint_pos_knee_L'])[:,0],cmd=o(S['command_x_all'])[:,0])
def brake_i(d):
    hi=d['cmd']>1.0; lo=d['cmd']<0.5
    return next(i for i in range(100,len(d['cmd'])-1) if hi[i-1] and lo[i])
def robot(ax,d,i,beta,refx,col,alpha):
    sh=lambda p:Rs(np.array(p)-np.array([refx,0]),beta)
    th=-d['pit'][i]; b,h,k,w=fk(d['px'][i],d['hz'][i],th,d['qh'][i],d['qk'][i]); B,H,K,W=sh(b),sh(h),sh(k),sh(w)
    z=4 if alpha<1 else 6
    ax.plot([H[0],K[0]],[H[1],K[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([K[0],W[0]],[K[1],W[1]],"-",c="#222",lw=3,zorder=z,solid_capstyle="round",alpha=alpha)
    ax.plot([B[0],H[0]],[B[1],H[1]],"-",c="#222",lw=2,zorder=z,alpha=alpha)
    ax.add_patch(plt.Circle(W,0.095,fc="#777",ec="k",zorder=z,alpha=alpha))
    bx=np.array([[-0.16,-0.05],[0.16,-0.05],[0.16,0.11],[-0.16,0.11]])
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in bx],closed=True,fc=col,ec="k",alpha=alpha*0.92,zorder=z))
    lx=np.array([[-0.06,0.11],[0.06,0.11],[0.06,0.23],[-0.06,0.23]])
    ax.add_patch(plt.Polygon([sh(b+roty(c,th)) for c in lx],closed=True,fc="#e0902a",ec="k",alpha=alpha,zorder=z))
    return W  # wheel pos (for arrow)
def panel(ax,d,iA,iB,beta,refx,col,xlim,ylim,label):
    sh=lambda p:Rs(np.array(p)-np.array([refx,0]),beta)
    gx=np.linspace(-2,7,60); pts=np.array([sh([x+refx,0]) for x in gx])
    ax.fill_between(pts[:,0],pts[:,1]-3,pts[:,1],color="#e8dcc8",zorder=0); ax.plot(pts[:,0],pts[:,1],"-",c="#8a6d3b",lw=1.6,zorder=1)
    O=sh([refx,0]); ax.plot([O[0],O[0]],[O[1],O[1]+0.6],"-",c="#444",lw=1.2,zorder=2)
    ax.add_patch(plt.Polygon([[O[0],O[1]+0.6],[O[0]+0.28,O[1]+0.5],[O[0],O[1]+0.4]],closed=True,fc="#e03b3b",ec="k",zorder=2))
    ax.text(O[0]-0.05,O[1]+0.62,"急停点",fontproperties=cjk,fontsize=9,ha="center",va="bottom",color="#444")
    WA=robot(ax,d,iA,beta,refx,col,0.32)   # before (faded)
    WB=robot(ax,d,iB,beta,refx,col,1.0)     # after (solid)
    # translation arrow A->B (overshoot)
    ax.add_patch(FancyArrowPatch((WA[0],WA[1]-0.18),(WB[0],WB[1]-0.18),arrowstyle="-|>",mutation_scale=22,color="#c0392b",lw=2.5,zorder=9))
    ax.text((WA[0]+WB[0])/2,min(WA[1],WB[1])-0.34,"%s %+.2f m"%(label,d['px'][iB]-d['px'][iA]),fontproperties=cjk,fontsize=11,ha="center",color="#c0392b",fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
M=load('qs1_resid1'); R=load('qs0_resid0_torq0'); beta=np.radians(-10); cM,cR="#2c6fbb","#d8551a"
ibM=brake_i(M); ibR=brake_i(R)
iAM,iBM=ibM,ibM+int(np.argmax(M['px'][ibM:ibM+150])); iAR,iBR=ibR,ibR+int(np.argmax(R['px'][ibR:ibR+150]))
xlim,ylim=(-0.9,4.3),(-1.5,0.9)
fig,axs=plt.subplots(2,1,figsize=(9.2,7.2))
panel(axs[0],M,iAM,iBM,beta,M['px'][ibM],cM,xlim,ylim,"刹停过点")
axs[0].text(-0.04,0.5,"Model-guided\n短刹停住",transform=axs[0].transAxes,fontproperties=cjk,fontsize=13,color=cM,fontweight="bold",ha="right",va="center")
panel(axs[1],R,iAR,iBR,beta,R['px'][ibR],cR,xlim,ylim,"冲过点")
axs[1].text(-0.04,0.5,"RL-only\n冲过头",transform=axs[1].transAxes,fontproperties=cjk,fontsize=13,color=cR,fontweight="bold",ha="right",va="center")
fig.suptitle("搭载 26 kg 下坡 10° 急停:刹车点→停止点(淡=刹车瞬间,实=停止)",fontproperties=cjk,fontsize=13.5,fontweight="bold")
plt.tight_layout(rect=[0.1,0,1,0.95]); plt.savefig(r"legged_gym/scripts/figs_ch5/paper_pdf/C10_急停刹车帧条.pdf"); plt.savefig("/tmp/_chk.png",dpi=130)
print("Model overshoot=%.2f RL overshoot=%.2f"%(M['px'][iBM]-M['px'][iAM],R['px'][iBR]-R['px'][iAR]))
