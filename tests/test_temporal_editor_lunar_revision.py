"""Regressões: latitude/longitude, títulos independentes e exportação legível."""
import tempfile
import unittest
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from iat_temporal_analysis import intervals, parse_metadata
from iat_temporal_lunar import location_settings
from iat_temporal_figure import export_style, make_class_palette, render_figure


class ExportAndLunarRevisionTests(unittest.TestCase):
    def setUp(self):
        names=[
            "2026-02-02_11_47_11.839_+0001537_x1702_y244.png",
            "2026-02-18_15_02_19.119_+0001537_x1702_y244.png",
            "2026-03-03_12_00_00.000_+0001537_x1702_y244.png",
        ]
        self.df=parse_metadata(pd.DataFrame({
            "names":names,"x":[-2.0,1.0,3.0],"y":[2.0,3.0,-2.0],
            "classe":["Grupo A","Grupo B","Grupo C"]}))
        self.settings={"points":True,"density":False,"contours":False}
        self.colors=make_class_palette(self.df["classe"].unique())
        self.panels=[("02/02",self.df.iloc[:1]),
                     ("18/02",self.df.iloc[1:2]),
                     ("03/03",self.df.iloc[2:])]

    def test_lunar_requires_site_fuso_and_uses_actual_conjunctions(self):
        with self.assertRaisesRegex(ValueError,"latitude"):
            intervals(self.df,"Data",period="Lunar (astronômico)")
        self.assertEqual(location_settings("-23.55","-46.63","-03:00"),
                         (-23.55,-46.63,-3.0))
        for site in [(100,0,0),(0,181,0),(0,0,15),(0,0,"")]:
            with self.assertRaises(ValueError):
                location_settings(*site)
        result=intervals(self.df,"Data","2026-02-01","2026-03-10","1",
                         period="Lunar (astronômico)",
                         lunar_site=(-23.55,-46.63,-3))
        self.assertEqual(sum((m.astype(int) for _,m in result)).tolist(),[1,1,1])
        self.assertTrue(any("17/02/2026" in name for name,_ in result))
        self.assertTrue(all("aprox" not in name for name,_ in result))

    def test_lunar_is_not_fixed_29_53_days_and_has_local_time(self):
        series=intervals(self.df,"Data","2026-02-01","2026-03-10","1",
                         period="Lunar (astronômico)",
                         lunar_site=(-23.55,-46.63,-3))
        self.assertGreaterEqual(len(series),2)
        # True Moon -> Sun conjunctions do not have artificially fixed length.
        d1=series[0][0].split(" → ")
        a=pd.to_datetime(d1[0].replace("Lua nova ",""),format="%d/%m/%Y %H:%M")
        b=pd.to_datetime(d1[1],format="%d/%m/%Y %H:%M")
        self.assertNotAlmostEqual((b-a).total_seconds()/86400,29.530588,places=4)

    def test_independent_titles_and_renamed_classes(self):
        style=export_style(panel_title_a="Antes",panel_title_b="Durante",
                          panel_title_c="Depois",legend_names={"Grupo A":"Classe experimental A"})
        fig=Figure(figsize=(25,8.5),dpi=80)
        axes=render_figure(fig,self.panels,(-4,6,-4,6),self.settings,self.colors,style)
        self.assertEqual([ax.get_title() for ax in axes],["Antes","Durante","Depois"])
        self.assertTrue(any("Classe experimental A" in tx.get_text()
                            for tx in fig.axes[-1].texts))

    def test_export_width_height_and_large_legends_expand_without_overlap(self):
        rows=pd.DataFrame({
            "classe":[f"Organismo experimental identificador longo número {i:02d}"
                      for i in range(15)],
            "x":np.arange(15.,dtype=float),"y":np.arange(15.,dtype=float)})
        palette=make_class_palette(rows["classe"])
        panels=[("Data início de campanha para monitorar",rows)]*3
        style=export_style(figure_width=13,figure_height=5,
                           legend_size=27,title_size=30,
                           panel_title_a="Tratamento experimental anterior",
                           panel_title_b="Após operação",
                           panel_title_c="Outra série")
        fig=Figure(figsize=(13,5),dpi=80)
        axes=render_figure(fig,panels,(-1,16,-1,16),self.settings,palette,style)
        self.assertGreater(fig.get_figheight(),5)
        self.assertGreater(fig.get_figwidth(),13)
        self.assertEqual(len(axes),3)
        canvas=FigureCanvasAgg(fig);canvas.draw()
        renderer=canvas.get_renderer()
        legend=fig.axes[-1]
        bbox=legend.get_window_extent(renderer=renderer)
        texts=legend.texts[1:]
        self.assertEqual(len(texts),15)
        last=0
        for text in reversed(texts):
            r=text.get_window_extent(renderer=renderer)
            self.assertLessEqual(r.x1,bbox.x1+5)
            self.assertGreaterEqual(r.x0,bbox.x0-5)
            self.assertGreaterEqual(r.y0,bbox.y0-10)
            self.assertGreaterEqual(r.y0,last-1)
            last=r.y1
        # Se a expansão for desmarcada, não deve produzir exportação cortada.
        with self.assertRaisesRegex(ValueError,"insuficiente"):
            render_figure(Figure(figsize=(13,5)),panels,(-1,16,-1,16),
                          self.settings,palette,{**style,"auto_expand":False})

    def test_very_large_independent_titles_remain_in_own_panels(self):
        style=export_style(
            panel_title_a="ANTES - AMOSTRAS DIURNAS PROFUNDIDADE CONSTANTE",
            panel_title_b="DURANTE - ACOMPANHAMENTO EXPERIMENTAL DO EQUIPAMENTO",
            panel_title_c="DEPOIS - DISTRIBUICAO FINAL DAS AMOSTRAS",
            title_size=34,figure_width=14,figure_height=5)
        fig=Figure(figsize=(14,5),dpi=90)
        axes=render_figure(fig,self.panels,(-4,6,-4,6),
                           self.settings,self.colors,style)
        FigureCanvasAgg(fig).draw()
        renderer=fig.canvas.get_renderer()
        boxes=[ax.title.get_window_extent(renderer=renderer) for ax in axes]
        for left,right in zip(boxes,boxes[1:]):
            self.assertLess(left.x1,right.x0)
        for rect in boxes:
            self.assertLess(rect.y1,fig.bbox.y1+2)

    def test_gif_frames_same_size_with_different_classes_and_long_labels(self):
        from PIL import Image
        from iat_temporal_figure import save_animated_gif
        frames=[
            ("Curto",pd.Series([True,False,False],index=self.df.index)),
            ("Outras classes e nomes diferentes 2026",pd.Series([False,True,True],
                                                           index=self.df.index))
        ]
        with tempfile.TemporaryDirectory() as folder:
            target=__import__("pathlib").Path(folder)/"teste.gif"
            save_animated_gif(target,self.df,frames,(-4,6,-4,6),
                              self.settings,self.colors,style=export_style(
                                  legend_size=17,figure_width=10,figure_height=5))
            with Image.open(target) as out:
                self.assertEqual(out.n_frames,2)
                size=out.size
                for i in range(out.n_frames):
                    out.seek(i)
                    self.assertEqual(out.size,size)


if __name__=="__main__":
    unittest.main()
