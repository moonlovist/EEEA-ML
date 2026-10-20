"""Mathematical and experimental checks, including all MLP parameters."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from models import half_mse_gradient, TinyMLP, Bigram, causal_attention
from ml_labs import split_data, read_rows, run_lab, ROOT
from feedback import check_answer

class MathematicalTests(unittest.TestCase):
    def test_half_mse_gradient_and_step(self):
        x,y=[1,2,3],[12,19,32]
        loss,dw,db=half_mse_gradient(x,y,10,0)
        self.assertEqual((loss,dw,db),(1.5,-2.,-1.))
        w,b=10-.1*dw,0-.1*db
        self.assertAlmostEqual(half_mse_gradient(x,y,w,b)[0],6.83/6)
        for which,expected in [(0,dw),(1,db)]:
            plus=[10.,0.];minus=plus.copy();plus[which]+=1e-5;minus[which]-=1e-5
            numerical=(half_mse_gradient(x,y,*plus)[0]-half_mse_gradient(x,y,*minus)[0])/2e-5
            self.assertAlmostEqual(numerical,expected,places=7)

    def test_mae_and_confusion(self):
        self.assertAlmostEqual(float(np.mean(np.abs(np.array([10,20,30])-np.array([12,19,32])))),5/3)
        matrix=np.array([[23,1],[3,21]])
        self.assertEqual(matrix.sum(),48)
        self.assertAlmostEqual(np.trace(matrix)/matrix.sum(),44/48)

    def test_all_mlp_derivatives(self):
        model=TinyMLP(width=8)
        X=np.array([[.2,-.5],[1.1,.7],[-.3,.4]])
        y=np.array([0.,1.,0.]);grads=model.gradients(X,y)
        count=0
        for name in ('W1','b1','W2','b2'):
            value=getattr(model,name)
            for index in np.ndindex(value.shape):
                old=value[index].copy();eps=1e-5
                value[index]=old+eps;plus=model.loss(X,y)
                value[index]=old-eps;minus=model.loss(X,y)
                value[index]=old
                numerical=(plus-minus)/(2*eps)
                self.assertAlmostEqual(float(grads[name][index]),numerical,places=7)
                count+=1
        self.assertEqual(count,33)

    def test_mlp_probability_and_update(self):
        model=TinyMLP(width=2);X=np.array([[0.,1.],[1.,0.]]);y=np.array([0.,1.])
        self.assertEqual(model.parameters,9)
        p=model.predict_proba(X)
        self.assertTrue(np.all((p>0)&(p<1)))
        initial=model.loss(X,y);model.step(X,y,.01)
        self.assertLess(model.loss(X,y),initial)

    def test_attention_rows_mask_and_value(self):
        q=np.array([[1.,0.],[0.,1.],[1.,1.]]);v=np.array([[10.,0.],[0.,10.],[10.,10.]])
        out,w=causal_attention(q,q,v)
        np.testing.assert_allclose(w.sum(axis=1),1)
        np.testing.assert_array_equal(w[np.triu_indices(3,1)],0)
        np.testing.assert_allclose(out[0],[10,0])
        a=1/(1+np.exp(1/np.sqrt(2)))
        np.testing.assert_allclose(out[1],[10*a,10*(1-a)])
        changed=v.copy();changed[2]=[999,999]
        np.testing.assert_allclose(causal_attention(q,q,changed)[0][:2],out[:2])

    def test_bigram_counts_temperature_and_budget(self):
        corpus=json.loads((ROOT/'data/corpus.json').read_text(encoding='utf-8'))
        for lang,current,word in [('en','a','note'),('fr','un','texte')]:
            m=Bigram(corpus[lang]);tokens,p=m.distribution(current)
            self.assertAlmostEqual(p[tokens.index(word)],4/7)
            tokens,p=m.distribution(current,.5)
            self.assertAlmostEqual(p[tokens.index(word)],16/25)
            self.assertLessEqual(len(m.generate(current,max_tokens=12).split()),13)
            self.assertEqual(m.generate(current,seed=1),m.generate(current,seed=1))
        with self.assertRaises(ValueError):m.distribution(current,0)

    def test_feedback(self):
        self.assertEqual(check_answer(None,1),'pending')
        self.assertEqual(check_answer(1,1),'correct')
        self.assertEqual(check_answer(0,1),'retry')

class ExperimentalTests(unittest.TestCase):
    def test_split_counts_and_identity(self):
        for filename,counts in [('walks.csv',(28,10,10)),('classification.csv',(96,48,48))]:
            sets=split_data(filename);rows=read_rows(filename)
            self.assertEqual(tuple(len(sets[k][1]) for k in ('train','validation','test')),counts)
            self.assertEqual(len(set(r['id'] for r in rows)),len(rows))
            for X,y in sets.values():self.assertEqual(len(X),len(y));self.assertTrue(np.isfinite(X).all())

    def test_all_labs_and_report_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in range(1,9):
                result=run_lab(n,'fr',Path(tmp)/str(n))
                self.assertTrue(Path(result['report_path']).exists())
                for filename in result['figures']:self.assertTrue((Path(result['report_path']).parent/filename).is_file())
                if n==6:
                    self.assertEqual(result['summary']['parameters'],33)
                    self.assertLess(result['summary']['final_training_BCE'],result['summary']['initial_training_BCE'])
                if n==8:self.assertEqual(result['summary']['test_count'],48)

    def test_reserved_test_values_leave_selection_unchanged(self):
        sets=split_data()
        changed={k:(X.copy(),y.copy()) for k,(X,y) in sets.items()}
        changed['test']=(np.full_like(sets['test'][0],999),1-sets['test'][1])
        with tempfile.TemporaryDirectory() as tmp:
            a=run_lab(4,output_dir=Path(tmp)/'a')
            with patch('ml_labs.split_data',return_value=changed):b=run_lab(4,output_dir=Path(tmp)/'b')
            self.assertEqual(a['summary'],b['summary'])
            self.assertEqual(a['tables'],b['tables'])

    def test_standardisation_uses_training_statistics(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        sets=split_data();X,y=sets['train']
        model=make_pipeline(StandardScaler(),KNeighborsClassifier(5)).fit(X,y)
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_,X.mean(axis=0))
        # Population convention and constant-column handling are explicit.
        scaler=StandardScaler().fit([[1.,5.],[3.,5.]])
        np.testing.assert_allclose(scaler.transform([[1.,5.],[3.,5.]]),[[-1,0],[1,0]])

    def test_kmeans_counts_and_original_unit_centres(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=run_lab(5,output_dir=tmp)
            for key,rows in r['tables'].items():
                if key.endswith('centres'):
                    self.assertEqual(sum(row['count'] for row in rows),90)
                    self.assertTrue(all(row['spend_eur_month']>0 for row in rows))

if __name__=='__main__':unittest.main()
