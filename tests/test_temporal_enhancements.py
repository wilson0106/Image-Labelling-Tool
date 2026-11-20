"""Cobertura das melhorias pontuais da Análise Temporal e Espacial."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from iat_temporal_analysis import parse_metadata, intervals
from iat_temporal_figure import (render_figure, legend_lines, export_style,
                                 make_class_palette, save_animated_gif)


class TemporalEnhancementTests(unittest.TestCase):
    def setUp(self):
        names=[
            "2024-02-28_10_00_00.000_+0001500_a.png",
            "2024-02-29_11_00_00.000_+0001500_b.png",
            "2024-03-01_10_00_00.000_+0001500_c.png",
            "2024-03-31_10_00_00.000_+0001500_d.png",
            "2024-04-01_10_00_00.000_+0001500_e.png",
        ]
        self.df=parse_metadata(pd.DataFrame({
            "names":names,"classe":["classe A"]*3+["classe B"]*2,
            "x":np.arange(5.,dtype=float),"y":np.arange(5.,dtype=float)}))
        self.palette=make_class_palette(self.df["classe"].unique())
        self.settings={"points":True,"density":False,"contours":False}

    def test_monthly_calendar_handles_leap_february_and_march(self):
        frames=intervals(self.df,"Data","2024-02-28","2024-04-01","1",
                         period="Mensal")
        self.assertEqual([lab for lab,_ in frames],["02/2024","03/2024","04/2024"])
        self.assertEqual([int(m.sum()) for _,m in frames],[2,2,1])
        self.assertEqual(sum((m.astype(int) for _,m in frames)).tolist(),[1]*5)

    def test_weekly_calendar_no_duplicate_dates(self):
        frames=intervals(self.df,"Data","2024-02-28","2024-04-01","1",
                         period="Semanal")
        self.assertEqual(sum((m.astype(int) for _,m in frames)).tolist(),[1]*5)

    def test_hourly_steps_by_real_date_and_hour(self):
        frames=intervals(self.df,"Data","2024-02-28","2024-02-28","1",
                         period="Horário")
        self.assertEqual(len(frames),24)
        self.assertEqual([int(mask.sum()) for _,mask in frames].count(1),1)

    def test_lunar_is_astronomical_and_bins_do_not_overlap(self):
        frames=intervals(self.df,"Data","2024-02-28","2024-04-01","1",
                         period="Lunar (astronômico)",
                         lunar_site=(-23.55,-46.63,-3))
        self.assertTrue(frames)
        self.assertTrue(all("Lua nova" in title for title,_ in frames))
        self.assertEqual(sum((m.astype(int) for _,m in frames)).tolist(),[1]*5)
        with self.assertRaisesRegex(ValueError,"latitude"):
            intervals(self.df,"Data","2024-02-28","2024-04-01","1",
                      period="Lunar (astronômico)")

    def test_comparison_three_panels_and_exported_legend(self):
        panels=[
            ("A",self.df.iloc[:2]),("B",self.df.iloc[2:4]),("C",self.df.iloc[4:])
        ]
        labels=dict(legend_lines(panels,self.palette,{"classe A":"Grupo Alfa"}))
        self.assertEqual(labels["classe A"],"Grupo Alfa (A: 2 | B: 1 | C: 0)")
        self.assertEqual(labels["classe B"],"classe B (A: 0 | B: 1 | C: 1)")
        fig=Figure(figsize=(21,8),dpi=100)
        style=export_style(title_template="Dia {intervalo}",
                           xlabel="Componente X",ylabel="Componente Y",
                           background="#FFFFFF",legend_names={"classe A":"Grupo Alfa"})
        axes=render_figure(fig,panels,(-2,7,-2,7),self.settings,self.palette,style)
        self.assertEqual(len(axes),3)
        self.assertEqual(len(fig.axes),4)
        self.assertEqual(fig.axes[0].get_title(),"Dia A")
        self.assertEqual(fig.axes[0].get_xlabel(),"Componente X")
        self.assertEqual(fig.axes[0].get_facecolor(),(1,1,1,1))
        texts=[x.get_text() for x in fig.axes[-1].texts]
        self.assertTrue(any("Grupo Alfa" in txt for txt in texts))
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"figura.pdf"
            fig.savefig(out,bbox_inches="tight")
            self.assertGreater(out.stat().st_size,3000)

    def test_gif_accepts_custom_white_style(self):
        frames=[("A",pd.Series(self.df.index<2,index=self.df.index)),
                ("B",pd.Series(self.df.index>=2,index=self.df.index))]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"custom.gif"
            save_animated_gif(path,self.df,frames,(-2,7,-2,7),self.settings,
                              self.palette,duration_ms=250,
                              style=export_style(background="#FFFFFF"))
            with Image.open(path) as image:
                self.assertEqual(image.n_frames,2)

    def test_invalid_period_and_step_rejected(self):
        with self.assertRaisesRegex(ValueError,"período"):
            intervals(self.df,"Data",period="Inexistente")
        with self.assertRaisesRegex(ValueError,"passo"):
            intervals(self.df,"Data",step="0",period="Mensal")


if __name__=="__main__":
    unittest.main()
