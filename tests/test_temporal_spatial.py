"""Regressões da subaba temporal/espacial: leitura somente e filtros combinados."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from iat_temporal_analysis import load_points, parse_metadata, intervals, filter_fixed

SAMPLES = [
    '2026-02-02_11_47_11.839_+0001537_estacaoA_x1702_y244.png',
    '2026-02-02_12_47_11.839_+0002537_estacaoB_x1702_y244.png',
    '2026-02-03_11_47_11.839_+0001537_estacaoA_x1702_y244.png',
    '2026-02-03_12_47_11.839_+0002537_estacaoB_x1702_y244.png',
]

class TemporalSpatialTests(unittest.TestCase):
    def setUp(self):
        self.frame = parse_metadata(pd.DataFrame({
            'names': SAMPLES, 'x': [0., 1., 2., 3.],
            'y': [1., 2., 3., 4.], 'classe': ['som', 'detritos', 'som', 'detritos'],
            'versao': ['v003'] * 4, 'batch_source': ['batch_01.csv']*4
        }))

    def test_extracts_date_time_depth(self):
        self.assertEqual(self.frame['depth'].tolist(), [1537,2537,1537,2537])
        self.assertEqual(self.frame['minute'].tolist(), [707,767,707,767])
        self.assertEqual(self.frame['day'].dt.strftime('%Y-%m-%d').tolist(),
                         ['2026-02-02','2026-02-02','2026-02-03','2026-02-03'])

    def test_daily_frames_share_coordinates(self):
        frames = intervals(self.frame,'Data','','','1')
        self.assertEqual([int(mask.sum()) for _,mask in frames], [2,2])
        self.assertEqual(self.frame[['x','y']].values.tolist(),
                         [[0.,1.],[1.,2.],[2.,3.],[3.,4.]])

    def test_hourly_and_depth_bins_do_not_overlap(self):
        for axis,start,end,step in [('Hora','11:00','13:00','60'),
                                    ('Profundidade','1000','3000','500')]:
            frames=intervals(self.frame,axis,start,end,step)
            total=sum(m.astype(int) for _,m in frames)
            self.assertEqual(total.tolist(),[1]*4)

    def test_custom_attribute_and_class_filter(self):
        frames=intervals(self.frame,'Atributo','','','1',r'_(estacao[A-Z])_')
        self.assertEqual([(label,int(mask.sum())) for label,mask in frames],
                         [('estacaoA',2),('estacaoB',2)])
        filt=filter_fixed(self.frame,'Data',
                          {'depth_min':'1000','depth_max':'2000',
                           'time_from':'11:00','time_to':'12:00'},
                          ['som'],'v003')
        self.assertEqual(len(filt),2)


    def test_plain_attribute_prefix_without_regex(self):
        frames=intervals(self.frame,'Atributo','','','1','estacao')
        self.assertEqual([(name,int(mask.sum())) for name,mask in frames],
                         [('estacaoA',2),('estacaoB',2)])
        with self.assertRaisesRegex(ValueError,'prefixo'):
            intervals(self.frame,'Atributo','','','1','')

    def test_no_version_restriction_when_showing_current_images(self):
        mixed=self.frame.copy()
        mixed.loc[0,'versao']='v004'
        chosen=filter_fixed(mixed,'Data',{},['som','detritos'])
        self.assertEqual(len(chosen),4)
        # Filtro textual de atributo não exige conhecimento de regex.
        chosen=filter_fixed(mixed,'Data',{'attribute_value':'estacaoB'},
                            ['som','detritos'])
        self.assertEqual(len(chosen),2)
        # Quando o atributo é o eixo, não se aplica o filtro textual oculto.
        chosen=filter_fixed(mixed,'Atributo',{'attribute_value':'estacaoB'},
                            ['som','detritos'])
        self.assertEqual(len(chosen),4)

    def test_empty_and_invalid_intervals(self):
        empty=filter_fixed(self.frame,'Data',{},[], 'v003')
        self.assertTrue(empty.empty)
        self.assertEqual(intervals(empty,'Data'),[])
        with self.assertRaises(ValueError):
            intervals(self.frame,'Profundidade','1000','3000','0')

    def test_project_loading_does_not_edit_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            file=folder/'batch_0001.csv'
            df=pd.DataFrame({'names':SAMPLES,'x':[0,1,2,3],
                             'y':[0,1,2,3],'manual_label':['som']*4,
                             'correct_label':['som']*4,
                             'feature_version':['v003']*4})
            df.to_csv(file,index=False)
            before=file.read_bytes()
            project=SimpleNamespace(csv_names=lambda:['batch_0001.csv'],
                                    dataframes_dir=folder,batch_cache={})
            points=load_points(project)
            self.assertEqual(len(points),4)
            self.assertEqual(before,file.read_bytes())

if __name__ == '__main__':
    unittest.main()
