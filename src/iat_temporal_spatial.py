"""Subaba independente de evolução temporal/espacial no IAT (somente leitura)."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
import queue
import threading
import pandas as pd
from iat_temporal_figure import density_reference, make_class_palette, render_figure, save_animated_gif, legend_lines
from iat_temporal_analysis import (PROJECTION_AXES, PROJECTION_METHODS, filter_fixed, intervals, load_points,
                                   temporal_projection)
import numpy as np
from iat_widgets import ScrollFrame
from ui_resources import IAT_COLORS as C

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    Figure=FigureCanvasTkAgg=None
INK='#FFFFFF'
SUB='#C9C9D1'
GRID='#2A2A30'
BG='#0B0B0D'
FONT=('Sans',10)

class TemporalSpatialPanel(tk.Frame):
    def __init__(self,parent,workspace):
        super().__init__(parent,bg=C['bg'])
        self.workspace=workspace
        self.data=pd.DataFrame()
        self.view=pd.DataFrame()
        self.frames=[]
        self.current=0
        self.timer=None
        self.bounds=None
        self.palette={}
        self._export_queue=None
        self._export_thread=None
        self._export_busy=False
        self.summary_visible=False
        self.vars={name:tk.StringVar(value=val) for name,val in {
            'axis':'Data','mode':'Individual','lower':'','upper':'','step':'1','axis_regex':'',
            'date_from':'','date_to':'','time_from':'','time_to':'',
            'depth_min':'','depth_max':'','attribute_regex':'','attribute_value':'',
            'version':'','frame_a':'','frame_b':'','frame_c':'',
             'period':'Diário','speed':'1 s',
             'lunar_lat':'','lunar_lon':'','lunar_tz':'',
            'caption':'Carregue a projeção do projeto para começar.',
            'frame_info':'—','projection':PROJECTION_METHODS['original'],
            'plot_title':'Projeção t-SNE fixa • mesmo referencial e eixos para todos os quadros'}.items()}
        # Coordinates shown: 'original' (CSV x/y) or a projection computed
        # from the embeddings (iat_temporal_analysis.temporal_projection).
        self.projection='original'
        self.coords={}
        self.load_token=0
        self.points=tk.BooleanVar(value=True)
        self.density=tk.BooleanVar(value=True)
        self.contours=tk.BooleanVar(value=False)
        # Density bands relative to a per-class peak shared by all frames (on)
        # or to each frame's own peak (off; frames are then not comparable).
        self.density_common=tk.BooleanVar(value=True)
        self.density_ref={}
        self.compare_three=tk.BooleanVar(value=False)
        self._build()
        self.bind('<Destroy>',self._destroy,add='+')
        # Lê dados do projeto em operação assíncrona apenas quando solicitado.

    def _destroy(self,event):
        if event.widget is self:self.pause()
    def label(self,parent,text,strong=False):
        return tk.Label(parent,text=text,bg=parent.cget('bg'),fg=INK if strong else SUB,
                        font=('Sans',10,'bold') if strong else FONT,anchor='w',justify='left')
    def entry(self,parent,key):
        e=tk.Entry(parent,textvariable=self.vars[key],bg=C['entry'],fg=INK,
                   insertbackground=INK,relief='flat',font=FONT,
                   highlightthickness=1,highlightbackground=C['border'],highlightcolor=C['blue'])
        e.pack(fill='x',pady=(0,5),ipady=3)
        return e
    def section(self,parent,title):
        heading=self.label(parent,title,True)
        heading.pack(anchor='w',pady=(12,4))
        return heading
    def line(self,parent,title,key):
        self.label(parent,title).pack(anchor='w')
        return self.entry(parent,key)
    def button(self,parent,text,callback,highlight=False):
        return tk.Button(parent,text=text,command=callback,relief='flat',bd=0,
                         fg='white',bg='#6D4FD8' if highlight else C['card2'],
                         activebackground='#5B3FC4' if highlight else C['border'],
                         activeforeground='white',highlightthickness=1,
                         highlightbackground='#6D4FD8' if highlight else C['border'],
                         font=('Sans',9,'bold' if highlight else 'normal'),
                         padx=10,pady=3,cursor='hand2')
    def _build(self):
        top=tk.Frame(self,bg=C['bg']);top.pack(fill='x',pady=(0,8))
        self.label(top,'Análise Temporal e Espacial',True).pack(side='left')
        self.button(top,'⟳ Carregar projeção',self.load,True).pack(side='right')
        body=tk.Frame(self,bg=C['bg']);body.pack(fill='both',expand=True)
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        left=ScrollFrame(body,bg=C['panel'])
        left.grid(row=0,column=0,sticky='nsew',padx=(0,8))
        left.configure(width=300);left.grid_propagate(False)
        p=left.inner
        q=tk.Frame(p,bg=C['panel']);q.pack(fill='both',expand=True,padx=10)
        self.section(q,'Projeção (coordenadas)')
        ttk.Combobox(q,values=list(PROJECTION_METHODS.values()),textvariable=self.vars['projection'],
                     state='readonly',style='Dark.TCombobox',font=FONT).pack(fill='x')
        self.button(q,'Usar / calcular projeção',self.use_projection).pack(fill='x',pady=(5,2))
        proj_tip=self.label(q,'Original = x/y do projeto. As demais são calculadas a partir dos embeddings '
                            'de todas as imagens carregadas e ficam em cache no projeto. UMAP supervisionado '
                            'usa os rótulos manuais: mostra a estrutura dada a anotação e não serve para '
                            'procurar erros de rótulo.')
        proj_tip.configure(wraplength=270);proj_tip.pack(fill='x')
        self.section(q,'Eixo de evolução')
        self.axis=ttk.Combobox(q,values=['Data','Hora','Profundidade','Atributo'],
                                textvariable=self.vars['axis'],state='readonly',style='Dark.TCombobox',font=FONT)
        self.axis.pack(fill='x');self.axis.bind('<<ComboboxSelected>>',self.axis_changed)
        self.axis_tip=self.label(q,'Data (AAAA-MM-DD), passo em dias.')
        self.axis_tip.pack(anchor='w')
        self.axis_range=tk.Frame(q,bg=C['panel'])
        self.axis_range.pack(fill='x')
        self.line(self.axis_range,'Intervalo inicial', 'lower')
        self.line(self.axis_range,'Intervalo final', 'upper')
        self.period_holder=tk.Frame(self.axis_range,bg=C['panel'])
        self.label(self.period_holder,'Unidade do intervalo (calendário)').pack(anchor='w')
        self.period_box=ttk.Combobox(self.period_holder,textvariable=self.vars['period'],
                values=['Horário','Diário','Semanal','Mensal','Trimestral','Anual',
                        'Lunar (astronômico)'],state='readonly',style='Dark.TCombobox',font=FONT)
        self.period_box.pack(fill='x',pady=(0,5))
        self.period_box.bind('<<ComboboxSelected>>',self._period_changed)
        self.lunar_site_panel=tk.Frame(self.axis_range,bg=C['panel'])
        self.label(self.lunar_site_panel,'Local da coleta • obrigatório no ciclo lunar',True).pack(anchor='w',pady=(5,3))
        self.line(self.lunar_site_panel,'Latitude (graus; Sul é negativo)','lunar_lat')
        self.line(self.lunar_site_panel,'Longitude (graus; Oeste é negativo)','lunar_lon')
        self.line(self.lunar_site_panel,'Fuso do horário no nome (UTC; ex.: -3)','lunar_tz')
        self.label(self.lunar_site_panel,
                   'Efeméride topocêntrica; quadros de Lua nova a Lua nova. '
                   'O fuso deve corresponder à data/hora das imagens.').pack(anchor='w')
        self.step_heading=self.label(self.axis_range,'Passo — quantidade de unidades')
        self.step_heading.pack(anchor='w')
        self.entry(self.axis_range,'step')
        self.axis_key_group=tk.Frame(q,bg=C['panel'])
        self.line(self.axis_key_group,'Prefixo do atributo no nome (ex.: estacao)','axis_regex')
        self.label(self.axis_key_group,'Ex.: estacaoA, estacaoB. Não precisa escrever regex.').pack(anchor='w')
        self.filter_heading=self.label(q,'Filtros adicionais — opcionais',True)
        self.filter_heading.pack(anchor='w',pady=(12,4))
        self.filter_tip=self.label(q,'Mantêm o mesmo recorte em todos os quadros.\nDeixe vazio para incluir todos os valores.')
        self.filter_tip.configure(wraplength=270)
        self.filter_tip.pack(fill='x',pady=(0,7))
        self.filter_groups={}
        for kind,fields,tip in (
            ('Data',[('De (AAAA-MM-DD)','date_from'),('Até (AAAA-MM-DD)','date_to')],
             'Escolha um período de coleta.'),
            ('Hora',[('De (HH:MM)','time_from'),('Até (HH:MM)','time_to')],
             'Ex.: 10:00 até 14:00, em todos os dias.'),
            ('Profundidade',[('Mínima (mm)','depth_min'),('Máxima (mm)','depth_max')],
             'Ex.: 1000 até 2000 mm.'),
            ('Atributo',[('Texto presente no nome da imagem','attribute_value')],
             'Ex.: estacaoB. Sem expressão regular.')):
            group=tk.Frame(q,bg=C['panel'])
            self.label(group,kind,True).pack(anchor='w',pady=(6,0))
            self.label(group,tip).pack(anchor='w')
            for title,key in fields:self.line(group,title,key)
            self.filter_groups[kind]=group
        self.classes_heading=self.section(q,'Classes de interesse')
        self.label(q,'Selecione uma ou várias classes (Ctrl+clique).').pack(anchor='w')
        classes_frame=tk.Frame(q,bg=C['panel'])
        classes_frame.pack(fill='x')
        self.classes=tk.Listbox(classes_frame,selectmode='multiple',exportselection=False,height=7,
                 bg=C['entry'],fg=INK,selectbackground='#3B5BDB',selectforeground='white',
                 font=FONT,highlightbackground=C['border'],highlightthickness=1,relief='flat',
                 yscrollcommand=lambda *args:self.classes_scroll.set(*args))
        self.classes.pack(side='left',fill='both',expand=True)
        self.classes_scroll=ttk.Scrollbar(classes_frame,orient='vertical',command=self.classes.yview)
        self.classes_scroll.pack(side='right',fill='y')
        controls=tk.Frame(q,bg=C['panel']);controls.pack(fill='x',pady=5)
        self.button(controls,'Todas',lambda:self.classes.select_set(0,'end')).pack(side='left')
        self.button(controls,'Nenhuma',lambda:self.classes.selection_clear(0,'end')).pack(side='left',padx=5)
        self.section(q,'Gráfico')
        for var,name in [(self.points,'Pontos individuais'),(self.density,'Manchas (≥ 20 imagens)'),
                         (self.density_common,'Escala de densidade comum a todos os quadros'),
                         (self.contours,'Contornos das manchas')]:
            tk.Checkbutton(q,text=name,variable=var,bg=C['panel'],fg=INK,
                           selectcolor=C['entry'],activebackground=C['panel'],highlightthickness=0,
                           activeforeground=INK,font=FONT).pack(anchor='w')
        self.button(q,'Aplicar e gerar intervalos',self.apply,True).pack(fill='x',pady=(11,14))

        # Rolagem própria mantém visíveis os controles inferiores mesmo em
        # janelas menores / escalas de texto maiores.
        self.center_scroll=ScrollFrame(body,bg=C['card'])
        self.center_scroll.grid(row=0,column=1,sticky='nsew')
        center=self.center_scroll.inner
        center.configure(bg=C['card'],highlightbackground=C['border'],highlightthickness=1)
        tk.Label(center,textvariable=self.vars['plot_title'],bg=center.cget('bg'),fg=INK,
                 font=('Sans',10,'bold'),anchor='w',justify='left').pack(anchor='w',padx=12,pady=(10,7))
        warn=self.label(center,'Compare somente batches com o mesmo referencial t-SNE e versão de rede.')
        warn.configure(wraplength=640)
        warn.pack(anchor='w',padx=12)
        modes=tk.Frame(center,bg=C['card']);modes.pack(fill='x',padx=10,pady=(7,3))
        self.summary_toggle=self.button(modes,'Mostrar resumo detalhado',self.toggle_summary)
        self.summary_toggle.pack(side='right',padx=(8,0))
        for mode in ['Individual','Comparar','Animação']:
            tk.Radiobutton(modes,text=mode,variable=self.vars['mode'],value=mode,
                command=self.on_mode,indicatoron=False,font=('Sans',9,'bold'),
                fg=INK,bg=C['card2'],selectcolor='#6D4FD8',
                activebackground='#5B3FC4',activeforeground='white',bd=0,
                highlightthickness=1,highlightbackground=C['border'],
                relief='flat',offrelief='flat',padx=14,pady=4).pack(side='left',padx=(0,4))
        self.fig=Figure(figsize=(9,5),dpi=100,facecolor=BG) if Figure else None
        if self.fig:
            self.canvas=FigureCanvasTkAgg(self.fig,master=center)
            widget=self.canvas.get_tk_widget()
            # Dark canvas background: no white band can show around the figure.
            widget.configure(height=500,bg=BG,highlightthickness=0)
            widget.pack(fill='x',padx=8,pady=(2,3))
            self._resize_job=None
            widget.bind('<Configure>',self.resize,add='+')
        else:self.label(center,'Matplotlib indisponível.').pack(expand=True)
        # Transporte, linha do tempo e salvamento têm linhas próprias para não se sobrepor.
        transport=tk.Frame(center,bg=C['card']);transport.pack(fill='x',padx=10,pady=(3,2))
        for txt,action in [('◀',lambda:self.next(-1)),('▶ Reproduzir',self.play),
                            ('❚❚ Pausar',self.pause),('▶',lambda:self.next(1))]:
            self.button(transport,txt,action).pack(side='left',padx=(0,5))
        ttk.Combobox(transport,textvariable=self.vars['speed'],width=7,state='readonly',style='Dark.TCombobox',
                     values=['0.5 s','1 s','2 s','3 s']).pack(side='right',padx=(4,0))
        self.label(transport,'Velocidade').pack(side='right')
        timeline=tk.Frame(center,bg=C['card']);timeline.pack(fill='x',padx=12,pady=(0,5))
        tk.Label(timeline,textvariable=self.vars['frame_info'],bg=C['card'],fg=INK,
                 font=('Sans',11,'bold'),anchor='e').pack(side='right',padx=(8,0))
        self.slider=ttk.Scale(timeline,from_=0,to=1,command=self.scrub)
        self.slider.pack(side='left',fill='x',expand=True,pady=6)
        export=tk.Frame(center,bg=C['card'])
        export.pack(fill='x',padx=10,pady=(0,6))
        export.columnconfigure(0,weight=1);export.columnconfigure(1,weight=1)
        self.button(export,'Salvar figura (PNG / SVG / PDF)',self.save_plot).grid(
            row=0,column=0,sticky='ew',padx=(0,5))
        self.gif_button=self.button(export,'Exportar animação (GIF)',self.save_gif)
        self.gif_button.grid(row=0,column=1,sticky='ew',padx=(5,0))
        self.gif_button.configure(state='disabled')
        self.export_status=tk.StringVar(value='')
        tk.Label(export,textvariable=self.export_status,bg=C['card'],fg=INK,
                 font=('Sans',10),anchor='w').grid(row=1,column=0,columnspan=2,sticky='ew')
        self.compare_controls=tk.Frame(center,bg=C['card'])
        side=self.compare_controls
        row=tk.Frame(side,bg=C['card']);row.pack(fill='x')
        self.label(row,'A:').pack(side='left')
        self.a=ttk.Combobox(row,width=17,textvariable=self.vars['frame_a'],state='readonly',style='Dark.TCombobox')
        self.a.pack(side='left',fill='x',expand=True,padx=(4,9))
        self.label(row,'B:').pack(side='left')
        self.b=ttk.Combobox(row,width=17,textvariable=self.vars['frame_b'],state='readonly',style='Dark.TCombobox')
        self.b.pack(side='left',fill='x',expand=True,padx=(4,0))
        self.c_holder=tk.Frame(side,bg=C['card'])
        self.label(self.c_holder,'C:').pack(side='left')
        self.c=ttk.Combobox(self.c_holder,width=24,
                            textvariable=self.vars['frame_c'],state='readonly',style='Dark.TCombobox')
        self.c.pack(side='left',fill='x',expand=True,padx=(4,0))
        tk.Checkbutton(side,text='Incluir terceiro intervalo (C)',
                       variable=self.compare_three,command=self._third_changed,
                       bg=C['card'],fg=INK,selectcolor=C['entry'],
                       activebackground=C['card'],activeforeground=INK,
                       font=FONT).pack(anchor='w',padx=6,pady=(5,1))
        self.button(side,'Atualizar comparação',self.draw).pack(anchor='e',pady=(4,0))
        # Resumo disponível sob demanda: não comprime os gráficos e suas legendas.
        self.summary_panel=tk.Frame(center,bg=C['panel'],highlightbackground=C['border'],
                                   highlightthickness=1)
        self.label(self.summary_panel,'Resumo por classe',True).pack(
            anchor='w',padx=10,pady=(7,0))
        summary_body=tk.Frame(self.summary_panel,bg=C['panel'])
        summary_body.pack(fill='both',expand=True,padx=8,pady=4)
        self.summary=tk.Text(summary_body,bg=C['panel'],fg=INK,font=FONT,wrap='word',
                             height=7,state='disabled',relief='flat')
        self.summary.pack(side='left',fill='both',expand=True)
        summary_scroll=ttk.Scrollbar(summary_body,orient='vertical',command=self.summary.yview)
        summary_scroll.pack(side='right',fill='y')
        self.summary.configure(yscrollcommand=summary_scroll.set)
        self.label(self.summary_panel,
            'Contagens de imagens, não abundância ambiental normalizada.').pack(
                anchor='w',padx=10,pady=(0,6))
        self.caption=tk.Label(self,bg=C['bg'],fg=INK,font=('Sans',10,'bold'),
                     textvariable=self.vars['caption'],anchor='w')
        self.caption.pack(fill='x',pady=(4,0))
        self.axis_changed()
        self._update_export_mode()

    def toggle_summary(self):
        self.summary_visible=not self.summary_visible
        if self.summary_visible:
            self.summary_panel.pack(fill='x',padx=10,pady=(0,8))
            self.summary_toggle.configure(text='Ocultar resumo detalhado')
        else:
            self.summary_panel.pack_forget()
            self.summary_toggle.configure(text='Mostrar resumo detalhado')

    def resize(self,event):
        # Matplotlib's own <Configure> handler already resized the figure using
        # the real DPI (the previous code assumed 100 dpi, leaving a white band
        # on HiDPI screens).  Only the legend wrapping needs a redraw, debounced
        # so dragging the window does not re-render on every pixel.
        if not self.fig or event.width<=220 or event.height<=200:
            return
        if self._resize_job is not None:
            try:self.after_cancel(self._resize_job)
            except (tk.TclError,RuntimeError):pass
        self._resize_job=self.after(120,self._after_resize)
    def _after_resize(self):
        self._resize_job=None
        if not self.winfo_exists():return
        if self.frames:
            self.draw()
        else:
            self.canvas.draw_idle()
    def _period_changed(self,event=None):
        period=self.vars['period'].get()
        self.lunar_site_panel.pack_forget()
        if period=='Lunar (astronômico)' and self.vars['axis'].get()=='Data':
            self.lunar_site_panel.pack(fill='x',pady=(2,6),before=self.step_heading)
            self.axis_tip.configure(
                text='Ciclos reais Lua nova → Lua nova. Informe local e fuso UTC das imagens.')
        elif self.vars['axis'].get()=='Data':
            self.axis_tip.configure(text='Intervalos de calendário. Mês = mês real, não 30 dias.')
    def _third_changed(self):
        if self.compare_three.get():
            self.c_holder.pack(fill='x',padx=6,pady=(1,3))
        else:
            self.c_holder.pack_forget()
        self.draw()
    def axis_changed(self,event=None):
        val=self.vars['axis'].get()
        self.axis_tip.configure(text={
            'Data':'Escolha Horário, Diário, Semanal, Lunar, Mensal, Trimestral ou Anual.',
            'Hora':'Cada quadro mostra uma faixa horária. Passo em minutos.',
            'Profundidade':'Cada quadro mostra uma faixa de profundidade. Passo em mm.',
            'Atributo':'Cada quadro mostra um valor do atributo (ex.: estacaoA, estacaoB).'
        }[val],wraplength=270)
        self.vars['step'].set({'Data':'1','Hora':'60','Profundidade':'500','Atributo':'1'}[val])
        self.period_holder.pack_forget()
        if val=='Data':
            self.period_holder.pack(fill='x',before=self.step_heading)
            self._period_changed()
        self.step_heading.configure(text='Passo — quantidade de unidades' if val=='Data'
                                    else ('Passo (minutos)' if val=='Hora'
                                          else 'Passo (mm)' if val=='Profundidade'
                                          else 'Passo'))
        for v in ('lower','upper'):self.vars[v].set('')
        # Os filtros não variam junto com o eixo selecionado: ocultar o filtro redundante.
        for name,group in self.filter_groups.items():
            group.pack_forget()
            if name!=val:group.pack(fill='x',pady=(0,3),before=self.classes_heading)
        self.axis_key_group.pack_forget()
        self.axis_range.pack_forget()
        if val=='Atributo':
            self.axis_key_group.pack(fill='x',before=self.filter_heading)
        else:
            self.axis_range.pack(fill='x',before=self.filter_heading)
        self.pause()
    def load(self):
        if self.workspace.project is None:
            messagebox.showinfo('Análise Temporal','Carregue um projeto no IAT.');return
        project=self.workspace.project
        self.vars['caption'].set('Carregando CSVs do projeto...')
        def done(frame):
            if not self.winfo_exists():return
            self.data=frame
            self.load_token+=1
            self.coords={'original':(frame['x'].to_numpy(float).copy(),frame['y'].to_numpy(float).copy())}
            self.projection='original'
            self.vars['projection'].set(PROJECTION_METHODS['original'])
            self._projection_title()
            self.palette=make_class_palette(frame["classe"].unique())
            old=set(self.classes.get(i) for i in self.classes.curselection())
            self.classes.delete(0,'end')
            for n,cl in enumerate(sorted(frame['classe'].unique())):
                self.classes.insert('end',cl)
                if not old or cl in old:self.classes.select_set(n)
            versions=sorted(frame['versao'].unique())
            note=(' • atenção: coordenadas de versões diferentes podem não ser comparáveis'
                  if len(versions)>1 else '')
            self.vars['caption'].set(f'{len(frame):,} imagens • usando resultados atuais{note}'.replace(',','.'))
            self.apply()
        self.workspace._run_operation('Carregando projeção temporal',
                  lambda progress:load_points(project,progress),done)

    def _projection_title(self):
        name=PROJECTION_METHODS.get(self.projection,self.projection)
        self.vars['plot_title'].set(f'Projeção: {name} • mesmo referencial e eixos para todos os quadros')

    def _use_coords(self,method):
        x,y=self.coords[method]
        self.data=self.data.assign(x=x,y=y)
        self.projection=method
        self._projection_title()

    def use_projection(self):
        """Switch the displayed coordinates; computes them in the background
        the first time a method is chosen (cached afterwards)."""
        if self.data.empty or self.workspace.project is None:
            messagebox.showinfo('Análise Temporal','Clique em Carregar projeção primeiro.');return
        label=self.vars['projection'].get()
        method=next((k for k,v in PROJECTION_METHODS.items() if v==label),'original')
        if method in self.coords:
            self._use_coords(method);self.apply();return
        project=self.workspace.project;data=self.data;token=self.load_token
        def done(res):
            if token!=self.load_token or not self.winfo_exists():return
            coords,note=res
            self.coords[method]=(coords[:,0].copy(),coords[:,1].copy())
            self._use_coords(method)
            self.apply()
            self.vars['caption'].set(note)
        self.workspace._run_operation('Projeção: '+label,
                                      lambda progress:temporal_projection(project,data,method,progress),done)

    def apply(self):
        self.pause()
        if self.data.empty:
            self.vars['caption'].set('Clique em Carregar projeção primeiro.')
            return
        try:
            keys=['date_from','date_to','time_from','time_to','depth_min','depth_max',
                  'attribute_value']
            params={k:self.vars[k].get().strip() for k in keys}
            selected=[self.classes.get(i) for i in self.classes.curselection()]
            self.view=filter_fixed(self.data,self.vars['axis'].get(),params,selected)
            if not self.view.empty:
                # Images without an embedding have no coordinates in a computed projection.
                finite=np.isfinite(self.view['x'].to_numpy(float))&np.isfinite(self.view['y'].to_numpy(float))
                if not finite.all():
                    self.view=self.view.loc[finite]
            # Colours are assigned over the classes actually displayed, so a
            # small selection gets the most distinct colours; they stay fixed
            # across all frames, comparison panels and exports of this view.
            if not self.view.empty:
                self.palette=make_class_palette(self.view['classe'].unique())
            use_lunar=(self.vars['axis'].get()=='Data'
                       and self.vars['period'].get()=='Lunar (astronômico)')
            site=(self.vars['lunar_lat'].get(),self.vars['lunar_lon'].get(),
                  self.vars['lunar_tz'].get()) if use_lunar else None
            self.frames=intervals(self.view,self.vars['axis'].get(),self.vars['lower'].get(),
                self.vars['upper'].get(),self.vars['step'].get(),self.vars['axis_regex'].get(),
                period=self.vars['period'].get() if self.vars['axis'].get()=='Data' else None,
                lunar_site=site)
            self.current=0
            labels=[lab for lab,_ in self.frames]
            self.a.configure(values=labels);self.b.configure(values=labels)
            self.c.configure(values=labels)
            self.vars['frame_a'].set(labels[0] if labels else '')
            self.vars['frame_b'].set(labels[-1] if labels else '')
            self.vars['frame_c'].set(labels[len(labels)//2] if labels else '')
            self.slider.configure(from_=0,to=max(len(self.frames)-1,0));self.slider.set(0)
            self.bounds=None
            if not self.view.empty:
                x=self.view['x'];y=self.view['y']
                dx=max(x.max()-x.min(),1)*.06;dy=max(y.max()-y.min(),1)*.06
                self.bounds=(x.min()-dx,x.max()+dx,y.min()-dy,y.max()+dy)
            # One reference peak per class for every frame (common density scale).
            self.density_ref=density_reference(self.view,self.frames,self.bounds)
            self.vars['caption'].set(
                f'{len(self.view):,} imagens • {len(self.frames)} intervalos • dados intactos'.replace(',','.'))
            self.draw()
        except (ValueError,TypeError) as exc:
            messagebox.showerror('Análise temporal',str(exc))

    def on_mode(self):
        self.pause()
        self._update_export_mode()
        self.draw()
    def _update_export_mode(self):
        if hasattr(self,'gif_button'):
            self.gif_button.configure(state='normal' if self.vars['mode'].get()=='Animação'
                                      and not self._export_busy else 'disabled')
        if hasattr(self,'compare_controls'):
            self.compare_controls.pack_forget()
            if self.vars['mode'].get()=='Comparar':
                pack_kwargs={'fill':'x','padx':10,'pady':(0,7)}
                if self.summary_visible:
                    pack_kwargs['before']=self.summary_panel
                self.compare_controls.pack(**pack_kwargs)

    def next(self,delta):
        if self.frames:
            self.current=(self.current+delta)%len(self.frames)
            self.slider.set(self.current);self.draw()
    def scrub(self,v):
        if self.frames:
            index=max(0,min(len(self.frames)-1,round(float(v))))
            if index!=self.current:
                self.current=index;self.draw()
    def play(self):
        if not self.frames:return
        self.pause()
        self.vars['mode'].set('Animação')
        self._update_export_mode()
        self._schedule()
    def pause(self):
        if self.timer is not None:
            try:self.after_cancel(self.timer)
            except (tk.TclError,RuntimeError):pass
            self.timer=None
    def _schedule(self):
        ms={'0.5 s':500,'1 s':1000,'2 s':2000,'3 s':3000}.get(self.vars['speed'].get(),1000)
        self.timer=self.after(ms,self._tick)
    def _tick(self):
        self.timer=None
        if not self.winfo_exists() or self.vars['mode'].get()!='Animação':return
        self.next(1);self._schedule()
    def frame_from_label(self,label,default):
        return next(((name,m) for name,m in self.frames if name==label),self.frames[default])
    def _settings(self):
        common=bool(self.density_common.get())
        xlabel,ylabel=PROJECTION_AXES.get(getattr(self,'projection','original'),('t-SNE 1','t-SNE 2'))
        return dict(points=bool(self.points.get()),density=bool(self.density.get()),
                    contours=bool(self.contours.get()),
                    density_common=common,density_ref=dict(self.density_ref) if common else {},
                    xlabel=xlabel,ylabel=ylabel)

    def _display_frames(self):
        if not self.frames:
            return []
        if self.vars['mode'].get()=='Comparar':
            selected=[self.frame_from_label(self.vars['frame_a'].get(),0),
                      self.frame_from_label(self.vars['frame_b'].get(),len(self.frames)-1)]
            if self.compare_three.get():
                selected.append(self.frame_from_label(
                    self.vars['frame_c'].get(),len(self.frames)//2))
            return selected
        return [self.frames[self.current]]

    def _display_panels(self):
        return [(label,self.view.loc[mask.fillna(False)])
                for label,mask in self._display_frames()]

    def draw(self):
        if not self.fig:
            return
        panels=self._display_panels()
        if not panels:
            render_figure(self.fig, [], self.bounds, self._settings(), self.palette)
            self.vars['frame_info'].set('—')
            self._stats([])
        else:
            render_figure(self.fig, panels, self.bounds, self._settings(), self.palette)
            self.vars['frame_info'].set(
                f'{self.current+1}/{len(self.frames)} • {self.frames[self.current][0]}')
            self._stats([(title,rows['classe'].value_counts(),len(rows))
                         for title,rows in panels])
        self.canvas.draw_idle()

    def save_plot(self):
        """Janela de edição e prévia: a dashboard permanece com fundo escuro."""
        if not self.fig or not self.frames:
            messagebox.showinfo('Exportar figura',
                'Primeiro carregue a projeção e gere os intervalos.')
            return
        from iat_temporal_export_ui import ExportDialog
        ExportDialog(self,gif=False)

    def _save_image_styled(self,path,style):
        panels=self._display_panels()
        labels=legend_lines(panels,self.palette,style.get('legend_names'))
        n=len(panels)
        export=Figure(figsize=(float(style["figure_width"]),float(style["figure_height"])),
                      dpi=140,facecolor=style["background"])
        render_figure(export,panels,self.bounds,self._settings(),self.palette,style=style)
        # dpi/transparência vêm do editor (web); a janela clássica usa 180 dpi.
        export.savefig(path,facecolor=style["background"],
                       bbox_inches='tight',pad_inches=.18,dpi=float(style.get("dpi",180)),
                       transparent=bool(style.get("transparent",False)))
        self.export_status.set('Figura salva com a aparência personalizada.')

    def save_gif(self):
        if self.vars['mode'].get()!='Animação':
            messagebox.showinfo('Exportar GIF','Selecione o modo Animação.')
            return
        if not self.frames:
            messagebox.showinfo('Exportar GIF',
                                'Primeiro carregue a projeção e gere os intervalos.')
            return
        if self._export_busy:return
        from iat_temporal_export_ui import ExportDialog
        ExportDialog(self,gif=True)

    def _start_export_gif(self,path,style):
        if self._export_busy:return
        self.pause()
        self._export_busy=True
        self._update_export_mode()
        self._export_queue=queue.Queue()
        frames=[(label,mask.copy()) for label,mask in self.frames]
        view=self.view
        bounds=self.bounds
        settings=self._settings()
        palette=self.palette.copy()
        style=style.copy()
        duration={'0.5 s':500,'1 s':1000,'2 s':2000,'3 s':3000}.get(
                  self.vars['speed'].get(),1000)
        def work():
            try:
                total=save_animated_gif(path,view,frames,bounds,settings,palette,
                      duration_ms=duration,style=style,
                      progress=lambda done,n:self._export_queue.put(('progress',done,n)))
                self._export_queue.put(('done',total,None))
            except Exception as exc:
                self._export_queue.put(('error',str(exc),None))
        self.export_status.set('Preparando GIF...')
        self._export_thread=threading.Thread(target=work,daemon=True)
        self._export_thread.start()
        self.after(150,self._poll_export)

    def _poll_export(self):
        if not self.winfo_exists():
            return
        completed=False
        while self._export_queue is not None:
            try:kind,a,b=self._export_queue.get_nowait()
            except queue.Empty:break
            if kind=='progress':
                self.export_status.set(f'Exportando GIF: {a}/{b} quadros')
            elif kind=='done':
                self.export_status.set(f'GIF salvo ({a} quadros).')
                completed=True
            elif kind=='error':
                self.export_status.set('Erro ao exportar GIF.')
                messagebox.showerror('Exportar GIF',a)
                completed=True
        if completed or (self._export_thread is not None and not self._export_thread.is_alive()):
            self._export_busy=False
            self._update_export_mode()
        elif self._export_busy:
            self.after(150,self._poll_export)

    def _stats(self,summaries):
        lines=[]
        for title,counts,total in summaries:
            lines.extend((title,f'{total:,} imagens'.replace(',','.'),''))
            for cl,n in counts.items():
                pct=100*n/total if total else 0
                lines.append(f'{cl}\n  {n:,}  ({pct:.1f}%)'.replace(',','.'))
            lines.append('')
        self.summary.configure(state='normal');self.summary.delete('1.0','end')
        self.summary.insert('end','\n'.join(lines) or 'Nenhum dado nos filtros.')
        self.summary.configure(state='disabled')
