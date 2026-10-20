"""Small CPU experiments with local data and saved plots. Ting Tan · EEEA · Oct 2026."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import csv
import html
import json
from time import perf_counter
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, silhouette_score
from models import TinyMLP, Bigram, half_mse_gradient, causal_attention

TITLES = {
 1: ('Data, variables and prediction', 'Données, variables et prédiction'),
 2: ('Linear regression and errors', 'Régression linéaire et erreurs'),
 3: ('Classification and mistakes', 'Classification et erreurs'),
 4: ('Generalisation and model selection', 'Généralisation et sélection du modèle'),
 5: ('Clustering and feature scales', 'Clustering et échelles des variables'),
 6: ('A small neural network', 'Un petit réseau de neurones'),
 7: ('Next-token generation and attention', 'Génération du prochain token et attention'),
 8: ('Interpret and justify a complete experiment', 'Interpréter et justifier une expérience complète'),
}


def tr(lang, en, fr):
    return en if lang == 'en' else fr


def read_rows(name):
    with (ROOT / 'data' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def dataset(name='classification.csv'):
    rows = read_rows(name)
    features = ['distance_km'] if name == 'walks.csv' else ['x1', 'x2']
    target = 'duration_min' if name == 'walks.csv' else 'label'
    X = np.array([[float(row[k]) for k in features] for row in rows])
    y = np.array([float(row[target]) for row in rows])
    if target == 'label': y = y.astype(int)
    return X, y, np.array([row['split'] for row in rows])


def split_data(name='classification.csv'):
    X, y, role = dataset(name)
    return {key: (X[role == key], y[role == key]) for key in ('train', 'validation', 'test')}


def boundary(ax, predictor, X, y, title, lang):
    a, b = np.meshgrid(np.linspace(X[:, 0].min() - .4, X[:, 0].max() + .4, 90),
                       np.linspace(X[:, 1].min() - .4, X[:, 1].max() + .4, 70))
    grid = np.column_stack([a.ravel(), b.ravel()])
    z = predictor(grid).reshape(a.shape)
    ax.contourf(a, b, z, levels=[-.5,.5,1.5], colors=['#bde0f6','#f6d4b9'], alpha=.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=25)
    ax.set(xlabel='x1', ylabel='x2', title=title)


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + '.png'), dpi=130)
    plt.close(fig)
    return name + '.png'


def finish(lab, lang, directory, summary, tables, figures, started):
    result = {'lab': lab, 'lang': lang, 'title': TITLES[lab][0 if lang=='en' else 1],
              'summary': summary, 'tables': tables, 'figures': figures,
              'elapsed_seconds': round(perf_counter()-started, 4),
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
    (directory / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    body = '<h1>' + html.escape(result['title']) + '</h1><p>Ting Tan · EEEA · Oct 2026</p>'
    body += '<p>'+tr(lang,'Measured reference run · fictional classroom data','Exécution de référence mesurée · données pédagogiques fictives')+'</p>'
    body += '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    for title, records in tables.items():
        body += '<h2>'+html.escape(title)+'</h2><table>'
        if records:
            headers=list(records[0]);body+='<tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'
            for row in records:body+='<tr>'+''.join('<td>'+html.escape(str(row[h]))+'</td>' for h in headers)+'</tr>'
            with (directory / (''.join(c if c.isalnum() else '_' for c in title)+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=headers,delimiter=';');writer.writeheader();writer.writerows(records)
        body+='</table>'
    for figure in figures:body+='<p><img alt="'+html.escape(result['title'])+'" src="'+figure+'"></p>'
    body+='<p>'+html.escape(str(result['versions']))+'</p>'
    document='<!doctype html><html lang="'+lang+'"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(result['title'])+'</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#20394d}pre{background:#edf4f8;padding:16px;white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbcbd7}img{max-width:100%;height:auto}</style><body>'+body+'</body></html>'
    (directory / 'report.html').write_text(document, encoding='utf-8')
    result['report_path']=str(directory / 'report.html')
    return result


def run_lab(lab, lang='en', output_dir=None, **options):
    if lab not in TITLES or lang not in ('en','fr'):raise ValueError('lab: 1..8; lang: en/fr')
    directory=Path(output_dir) if output_dir else ROOT / 'outputs' / lang / ('lab%02d'%lab)
    directory.mkdir(parents=True,exist_ok=True)
    start=perf_counter(); tables={}; figures=[]; summary={}
    if lab==1:
        X,y,roles=dataset('walks.csv')
        tables[tr(lang,'First six journeys','Six premiers trajets')]=[r for r in read_rows('walks.csv') if r['split']=='train'][:6]
        summary={'rows':len(y),'input':'distance_km','target':'duration_min','split_counts':{r:int(sum(roles==r)) for r in ('train','validation','test')},'mean_distance_km':round(float(X.mean()),3)}
        fig,ax=plt.subplots(figsize=(7,4))
        ax.scatter(X[roles=='train',0],y[roles=='train'],color='#2171a5')
        ax.set(xlabel=tr(lang,'Distance (km)','Distance (km)'),ylabel=tr(lang,'Duration (min)','Durée (min)'),title=tr(lang,'Training journeys: one point per observation','Trajets d’entraînement : un point par observation'))
        figures.append(save_figure(fig,directory,'data'))
    elif lab==2:
        sets=split_data('walks.csv');X,y=sets['train'];Xv,yv=sets['validation']
        model=LinearRegression().fit(X,y);pred=model.predict(Xv)
        w,b=float(options.get('slope',10)),float(options.get('intercept',1))
        manual=w*Xv[:,0]+b
        summary={'slope_min_per_km':round(float(model.coef_[0]),6),'intercept_min':round(float(model.intercept_),6),
                 'validation_MAE_min':round(float(mean_absolute_error(yv,pred)),6),
                 'baseline_training_mean_MAE_min':round(float(mean_absolute_error(yv,np.full(len(yv),y.mean()))),6),
                 'manual_slope':w,'manual_intercept':b,'manual_validation_MAE_min':round(float(mean_absolute_error(yv,manual)),6)}
        tables[tr(lang,'Validation predictions','Prédictions de validation')]=[{'x_km':float(x),'y_min':float(t),'prediction_min':round(float(p),3),'absolute_error_min':round(float(abs(p-t)),3)} for x,t,p in zip(Xv[:,0],yv,pred)]
        fig,ax=plt.subplots(figsize=(7,4));ax.scatter(X[:,0],y,label=tr(lang,'Training','Entraînement'));ax.scatter(Xv[:,0],yv,marker='x',label=tr(lang,'Validation','Validation'))
        grid=np.linspace(0,5.3,100);ax.plot(grid,model.predict(grid[:,None]),label=tr(lang,'Fitted line','Droite ajustée'));ax.plot(grid,w*grid+b,'--',label=tr(lang,'Manual line','Droite manuelle'))
        ax.set(xlabel='km',ylabel='min');ax.legend();figures.append(save_figure(fig,directory,'regression'))
    elif lab==3:
        sets=split_data();X,y=sets['train'];Xv,yv=sets['validation']
        k=int(options.get('neighbors',5));depth=int(options.get('depth',3))
        if not 1<=k<=len(X) or not 1<=depth<=20:raise ValueError('neighbors: 1..96; depth: 1..20')
        models={'kNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k)),
                'Tree':DecisionTreeClassifier(max_depth=depth,random_state=7)}
        fig,axes=plt.subplots(1,2,figsize=(10,4));rows=[]
        for ax,(name,model) in zip(axes,models.items()):
            model.fit(X,y);pred=model.predict(Xv);cm=confusion_matrix(yv,pred,labels=[0,1])
            rows.append({'model':name,'training_accuracy':round(float(model.score(X,y)),4),'validation_accuracy':round(float(accuracy_score(yv,pred)),4)})
            summary[name+'_confusion_true_rows_predicted_columns']=cm.tolist()
            mistakes=np.flatnonzero(pred!=yv)
            tables[name+' '+tr(lang,'mistakes','erreurs')]=[{'validation_index':int(i),'x1':float(Xv[i,0]),'x2':float(Xv[i,1]),'true':int(yv[i]),'predicted':int(pred[i])} for i in mistakes[:6]]
            boundary(ax,model.predict,Xv,yv,(tr(lang,'Tree','Arbre') if name=='Tree' else name)+' · '+tr(lang,'validation points','points de validation'),lang)
        summary.update(neighbors=k,depth=depth,validation_size=len(yv));tables['Scores']=rows
        figures.append(save_figure(fig,directory,'classification'))
    elif lab in (4,8):
        sets=split_data();X,y=sets['train'];Xv,yv=sets['validation'];rows=[];models=[]
        depths=[1,2,3,5,10,None]
        min_leaf=int(options.get('min_leaf',1))
        if not 1<=min_leaf<=30:raise ValueError('min_leaf: 1..30')
        for d in depths:
            m=DecisionTreeClassifier(max_depth=d,min_samples_leaf=min_leaf,random_state=7).fit(X,y);models.append(m)
            rows.append({'depth':str(d) if d else 'full','training_accuracy':round(float(m.score(X,y)),6),'validation_accuracy':round(float(m.score(Xv,yv)),6),'leaves':int(m.get_n_leaves())})
        # First maximum: deterministic tie rule chooses the shallower candidate.
        best=max(range(len(models)),key=lambda i:models[i].score(Xv,yv))
        summary={'min_samples_leaf':min_leaf,'selected_depth':rows[best]['depth'],'selection_rule':'highest validation accuracy; shallower candidate wins a tie',
                 'baseline_majority_validation_accuracy':float(np.mean(yv==np.argmax(np.bincount(y))))}
        if lab==8:
            Xt,yt=sets['test'];pred=models[best].predict(Xt)
            summary.update(test_accuracy=float(accuracy_score(yt,pred)),test_count=len(yt),test_confusion_true_rows_predicted_columns=confusion_matrix(yt,pred,labels=[0,1]).tolist())
        tables[tr(lang,'Depth comparison','Comparaison des profondeurs')]=rows
        fig,ax=plt.subplots(figsize=(7,4));positions=np.arange(len(rows));ax.plot(positions,[r['training_accuracy'] for r in rows],'o-',label=tr(lang,'Training','Entraînement'));ax.plot(positions,[r['validation_accuracy'] for r in rows],'o-',label='Validation');ax.set_xticks(positions,[tr(lang,'full','libre') if r['depth']=='full' else r['depth'] for r in rows]);ax.set(xlabel=tr(lang,'Maximum depth','Profondeur maximale'),ylabel=tr(lang,'Accuracy','Exactitude'),ylim=(0,1.05));ax.legend();figures.append(save_figure(fig,directory,'generalisation'))
    elif lab==5:
        data=read_rows('customers.csv');X=np.array([[float(r['visits_month']),float(r['spend_eur_month'])] for r in data]);k=int(options.get('clusters',3))
        if not 2<=k<=6:raise ValueError('clusters: 2..6')
        scaler=StandardScaler().fit(X);Xs=scaler.transform(X)
        fig,axes=plt.subplots(1,2,figsize=(10,4));rows=[]
        for ax,name,Z in [(axes[0],'raw',X),(axes[1],'standardized',Xs)]:
            model=KMeans(n_clusters=k,random_state=7,n_init=10).fit(Z)
            centers=model.cluster_centers_ if name=='raw' else scaler.inverse_transform(model.cluster_centers_)
            rows.append({'space':name,'k':k,'inertia_in_this_space':round(float(model.inertia_),3),'silhouette_in_this_space':round(float(silhouette_score(Z,model.labels_)),4)})
            tables[name+' '+tr(lang,'centres','centres')]=[{'cluster':i,'visits_month':round(float(c[0]),2),'spend_eur_month':round(float(c[1]),2),'count':int(sum(model.labels_==i))} for i,c in enumerate(centers)]
            ax.scatter(X[:,0],X[:,1],c=model.labels_,cmap='viridis');ax.scatter(centers[:,0],centers[:,1],marker='X',s=160,c='red')
            ax.set(xlabel=tr(lang,'Visits / month','Visites / mois'),ylabel=tr(lang,'EUR / month','EUR / mois'),title=tr(lang,'Raw scale' if name=='raw' else 'Standardized scale','Échelle brute' if name=='raw' else 'Échelle standardisée'))
        tables['Scores']=rows;summary={'customers':len(X),'clusters':k,'mean':scaler.mean_.tolist(),'scale_population_sd':scaler.scale_.tolist(),'comparison':'Compare inertia within a fixed feature space.'};figures.append(save_figure(fig,directory,'clustering'))
    elif lab==6:
        sets=split_data();X,y=sets['train'];Xv,yv=sets['validation'];scaler=StandardScaler().fit(X);Xs=scaler.transform(X);Xvs=scaler.transform(Xv)
        width=int(options.get('width',8));rate=float(options.get('rate',.1));epochs=int(options.get('epochs',300))
        if not 1<=width<=32 or not .0001<=rate<=2 or not 1<=epochs<=1000:raise ValueError('width 1..32; rate .0001..2; epochs 1..1000')
        model=TinyMLP(width=width);history=[]
        for epoch in range(epochs+1):
            history.append({'epoch':epoch,'training_BCE':model.loss(Xs,y),'validation_BCE':model.loss(Xvs,yv)})
            if epoch<epochs:model.step(Xs,y,rate)
        summary={'width':width,'learning_rate':rate,'epochs':epochs,'parameters':model.parameters,
                 'initial_training_BCE':history[0]['training_BCE'],'final_training_BCE':history[-1]['training_BCE'],
                 'validation_accuracy':float(np.mean(model.predict(Xvs)==yv)),'scaler_fitted_rows':len(X)}
        tables[tr(lang,'Loss checkpoints','Étapes de la perte')]=[history[i] for i in sorted(set([0,epochs//4,epochs//2,epochs]))]
        with (directory/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([h['epoch'] for h in history],[h['training_BCE'] for h in history],label=tr(lang,'Training','Entraînement'));axes[0].plot([h['epoch'] for h in history],[h['validation_BCE'] for h in history],label='Validation');axes[0].set(xlabel=tr(lang,'Full-batch update','Mise à jour sur tout le lot'),ylabel=tr(lang,'Mean binary cross-entropy','Entropie croisée binaire moyenne'));axes[0].legend()
        boundary(axes[1],lambda grid:model.predict(scaler.transform(grid)),Xv,yv,'MLP · validation',lang);figures.append(save_figure(fig,directory,'neural_network'))
    elif lab==7:
        corpus=json.loads((ROOT/'data/corpus.json').read_text(encoding='utf-8'))[lang];model=Bigram(corpus);temperature=float(options.get('temperature',1.));token='a' if lang=='en' else 'un';tokens,probs=model.distribution(token,temperature)
        tables[tr(lang,'Next-token probabilities','Probabilités du prochain token')]=[{'context':token,'token':t,'probability':round(float(p),6)} for t,p in zip(tokens,probs)]
        q=np.array([[1.,0.],[0.,1.],[1.,1.]]);v=np.array([[10.,0.],[0.,10.],[10.,10.]])
        output,weights=causal_attention(q,q,v)
        tables[tr(lang,'Generations','Générations')]=[{'seed':seed,'temperature':temperature,'text':model.generate('the' if lang=='en' else 'le',temperature,seed=seed)} for seed in (1,2,3)]
        summary={'sentences':len(corpus),'temperature':temperature,'attention_weights':weights.tolist(),'attention_output':output.tolist(),'generation_model':'word-level empirical bigram','attention_example':'fixed illustrative Q, K, V; a separate calculation'}
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(tokens,probs);axes[0].set(ylim=(0,1),ylabel=tr(lang,'Probability','Probabilité'),title=token+' → ?');im=axes[1].imshow(weights,vmin=0,vmax=1,cmap='Blues');axes[1].set(xlabel=tr(lang,'Key position','Position clé'),ylabel=tr(lang,'Query position','Position requête'),xticks=[0,1,2],yticks=[0,1,2]);fig.colorbar(im,ax=axes[1]);figures.append(save_figure(fig,directory,'generation_attention'))
    return finish(lab,lang,directory,summary,tables,figures,start)


def show(result):
    """Notebook display with embedded figures and plain-text metrics."""
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
    from IPython.display import Image, display
    parent=Path(result['report_path']).parent
    for filename in result['figures']:display(Image(filename=str(parent/filename)))
