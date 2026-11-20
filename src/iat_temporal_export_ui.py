"""Editor de exportação de figuras da Análise Temporal e Espacial.

Exclusivo desta subaba. Não altera dados, projeções ou cores exibidas na dashboard.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure
from iat_temporal_figure import export_style, render_figure, legend_lines


_FONT_CHOICES=("DejaVu Sans","DejaVu Serif","Liberation Sans",
               "Liberation Serif","Arial","Times New Roman","monospace")


class ExportDialog(tk.Toplevel):
    """Prévia fiel da figura que será salva, com aparência independente da GUI."""
    def __init__(self, owner, gif=False):
        super().__init__(owner)
        self.owner=owner
        self.gif=bool(gif)
        self.title("Prévia e personalização da exportação" +
                   (" — GIF" if gif else " — PNG / SVG / PDF"))
        self.configure(bg="#1a1a1e")
        self.geometry("1260x780")
        self.minsize(900,600)
        self.transient(owner)
        self._vars={}
        initial=export_style()
        n=max(1,len(owner._display_panels()))
        initial["figure_width"]={1:16.5,2:20.0,3:25.0}.get(n,16.5)
        initial["figure_height"]=8.5
        # Axis titles follow the projection in use (t-SNE, UMAP, PCA…).
        try:
            current=owner._settings()
            initial["xlabel"]=current.get("xlabel",initial["xlabel"])
            initial["ylabel"]=current.get("ylabel",initial["ylabel"])
        except Exception:
            pass
        self.auto_expand=tk.BooleanVar(self,value=True)
        for k,v in initial.items():
            if k not in ("legend_names","auto_expand"):
                self._vars[k]=tk.StringVar(self,value=str(v))
        self.alias_vars={}
        self.aliases=None
        self._preview_timer=None
        self._full_preview=None
        self._photo=None
        self._build()
        self.after(250,self.refresh_preview)

    def _label(self, parent, text):
        ttk.Label(parent,text=text).pack(anchor="w",padx=6,pady=(5,1))

    def _entry(self,parent,key,help_text=None):
        self._label(parent,help_text or key)
        w=ttk.Entry(parent,textvariable=self._vars[key])
        w.pack(fill="x",padx=6,pady=(0,4))
        return w

    def _font_group(self,parent,title,prefix,with_title=False):
        box=ttk.LabelFrame(parent,text=title,padding=7)
        box.pack(fill="x",padx=8,pady=5)
        row=ttk.Frame(box);row.pack(fill="x")
        ttk.Label(row,text="Fonte").grid(row=0,column=0,sticky="w")
        families=ttk.Combobox(row,textvariable=self._vars[prefix+"_family"],
                              values=_FONT_CHOICES,state="readonly",width=20)
        families.grid(row=1,column=0,sticky="ew",padx=(0,6))
        ttk.Label(row,text="Tamanho (pt)").grid(row=0,column=1,sticky="w")
        ttk.Spinbox(row,from_=6,to=36,increment=1,textvariable=self._vars[prefix+"_size"],
                    width=7).grid(row=1,column=1,padx=6)
        ttk.Button(row,text="Cor da fonte",command=lambda:self._pick_color(prefix+"_color")).grid(
                   row=1,column=2,padx=6)
        row.columnconfigure(0,weight=1)
        if with_title:
            row2=ttk.Frame(box);row2.pack(fill="x",pady=(4,0))
            ttk.Label(row2,text="Tamanho do cabeçalho").pack(side="left")
            ttk.Spinbox(row2,from_=6,to=36,textvariable=self._vars["legend_title_size"],
                        width=6).pack(side="left",padx=6)
            ttk.Button(row2,text="Cor do cabeçalho",command=lambda:self._pick_color(
                "legend_title_color")).pack(side="left")

    def _pick_color(self,key):
        val=self._vars[key].get()
        color=colorchooser.askcolor(color=val,parent=self,title="Selecionar cor")[1]
        if color:
            self._vars[key].set(color)
            self.refresh_preview()

    def _build(self):
        heading=tk.Frame(self,bg="#1a1a1e")
        heading.pack(fill="x",padx=12,pady=10)
        tk.Label(heading,text="PERSONALIZAR FIGURA",font=("Sans",14,"bold"),
                 bg="#1a1a1e",fg="#F4F8FF").pack(side="left")
        tk.Label(heading,text="Prévia antes de salvar • padrão: fundo branco, letras pretas",
                 bg="#1a1a1e",fg="#e9e9eb").pack(side="right")
        body=ttk.Panedwindow(self,orient="horizontal")
        body.pack(fill="both",expand=True,padx=12,pady=(0,8))
        left=ttk.Frame(body,width=360);body.add(left,weight=1)
        right=ttk.Frame(body);body.add(right,weight=3)
        tabs=ttk.Notebook(left)
        tabs.pack(fill="both",expand=True)
        names=ttk.Frame(tabs);fonts=ttk.Frame(tabs);background=ttk.Frame(tabs);classes=ttk.Frame(tabs)
        tabs.add(names,text="Textos")
        tabs.add(fonts,text="Fontes")
        tabs.add(background,text="Fundo")
        tabs.add(classes,text="Classes")
        dimensions=ttk.Frame(tabs)
        tabs.add(dimensions,text="Dimensões")
        self._entry(names,"title_template",
                    "Título padrão (use {intervalo} e {painel} se desejar)")
        ttk.Label(names,text="Títulos individuais (vazio = título padrão). "
                  "Cada campo controla somente o painel indicado.",
                  wraplength=290).pack(anchor="w",padx=6,pady=(6,4))
        for suffix in ("a","b","c"):
            self._entry(names,"panel_title_"+suffix,
                        "Título do gráfico "+suffix.upper())
        self._entry(names,"xlabel","Título do eixo horizontal")
        self._entry(names,"ylabel","Título do eixo vertical")
        self._entry(names,"legend_title","Título da legenda")
        self._label(names,"No modo Comparar, cada painel recebe seu próprio título.")
        self._font_group(fonts,"Título dos gráficos","title")
        self._font_group(fonts,"Títulos dos eixos","axis")
        self._font_group(fonts,"Números dos eixos","tick")
        self._font_group(fonts,"Itens da legenda","legend",with_title=True)
        ttk.Label(background,text="Cor de fundo da figura (inclusive a área dos eixos)").pack(
                  anchor="w",padx=9,pady=(18,7))
        self._entry(background,"background","Cor hexadecimal (ex.: #FFFFFF)")
        ttk.Button(background,text="Escolher cor…",
                   command=lambda:self._pick_color("background")).pack(anchor="w",padx=8)
        ttk.Label(background,text="A cor padrão é branca; títulos, números e legenda são pretos.",
                  wraplength=285).pack(anchor="w",padx=8,pady=14)
        self._label(classes,"Renomear as classes apenas na exportação")
        ttk.Label(classes,text="Classes do projeto à esquerda; novos nomes à direita. "
                  "Não altera os CSVs nem as cores.",
                  wraplength=290).pack(anchor="w",padx=8,pady=6)
        wrap=ttk.Frame(classes);wrap.pack(fill="both",expand=True,padx=7,pady=5)
        canvas=tk.Canvas(wrap,bg="white",highlightthickness=0,height=360)
        scroll=ttk.Scrollbar(wrap,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y")
        canvas.pack(side="left",fill="both",expand=True)
        entries=ttk.Frame(canvas)
        window=canvas.create_window((0,0),window=entries,anchor="nw")
        entries.bind("<Configure>",
            lambda e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e:canvas.itemconfigure(window,width=e.width))
        ttk.Label(entries,text="Classe original",font=("Sans",10,"bold")).grid(
            row=0,column=0,sticky="w",padx=(3,4),pady=4)
        ttk.Label(entries,text="Nome na exportação",font=("Sans",10,"bold")).grid(
            row=0,column=1,sticky="w",padx=(3,4),pady=4)
        entries.columnconfigure(1,weight=1)
        if self.gif and hasattr(self.owner,"view") and not self.owner.view.empty:
            visible_names=set(self.owner.view["classe"].dropna().astype(str))
        else:
            visible_names={str(name) for _,rows in self.owner._display_panels()
                           for name in rows["classe"].dropna().unique()}
        visible_names=visible_names & set(self.owner.palette)
        if not visible_names:
            visible_names=set(self.owner.palette)
        for index,name in enumerate(sorted(visible_names,key=str.casefold),start=1):
            var=tk.StringVar(self,value=name)
            self.alias_vars[name]=var
            ttk.Label(entries,text=name,wraplength=125).grid(
                row=index,column=0,sticky="w",padx=(3,6),pady=3)
            ttk.Entry(entries,textvariable=var,width=24).grid(
                row=index,column=1,sticky="ew",padx=(2,4),pady=3)

        self._label(dimensions,"Dimensões da figura para exportar")
        self._entry(dimensions,"figure_width","Largura (polegadas)")
        self._entry(dimensions,"figure_height","Altura (polegadas)")
        ttk.Checkbutton(dimensions,text="Expandir automaticamente para não "
                        "sobrepor títulos ou legenda",
                        variable=self.auto_expand).pack(anchor="w",padx=8,pady=(10,8))
        ttk.Label(dimensions,wraplength=290,text=
                  "Prévia e arquivo com as mesmas dimensões. Quando necessário, "
                  "a figura aumenta sem reduzir os textos. 1 pol = 2,54 cm.").pack(
                       anchor="w",padx=8,pady=8)
        # A prévia exibe exatamente o mesmo renderizador usado para salvar.
        ttk.Label(right,text="PRÉVIA DE EXPORTAÇÃO (dados e contagens reais do quadro)",
                  font=("Sans",11,"bold")).pack(anchor="w",padx=8,pady=(4,6))
        self.figure=Figure(figsize=(12.5,7.1),dpi=100,facecolor="white")
        self.preview=tk.Canvas(right,bg="white",highlightthickness=0)
        self.preview.pack(fill="both",expand=True,padx=8,pady=4)
        self.preview.bind("<Configure>",self._resize,add="+")
        actions=ttk.Frame(right);actions.pack(fill="x",padx=8,pady=8)
        ttk.Button(actions,text="Atualizar prévia",command=self.refresh_preview).pack(
            side="left",padx=(0,6))
        ttk.Button(actions,text="Ampliar prévia",command=self._zoom_preview).pack(
            side="left",padx=(0,6))
        ttk.Button(actions,text="Restaurar padrão",command=self._reset).pack(
            side="left",padx=(0,6))
        ttk.Button(actions,text="Cancelar",command=self.destroy).pack(side="right",padx=(6,0))
        ttk.Button(actions,text="Salvar GIF…" if self.gif else "Salvar figura…",
                   command=self._save).pack(side="right")
        self.status=tk.StringVar(value="")
        ttk.Label(right,textvariable=self.status).pack(anchor="w",padx=8,pady=(0,5))

    def _reset(self):
        for k,v in export_style().items():
            if k in self._vars:self._vars[k].set(v)
        self.auto_expand.set(True)
        for name,var in self.alias_vars.items():var.set(name)
        self.refresh_preview()

    def _resize(self,event):
        if event.width>350 and event.height>250:
            if self._preview_timer:
                self.after_cancel(self._preview_timer)
            self._preview_timer=self.after(200,self.refresh_preview)

    def _zoom_preview(self):
        if self._full_preview is None:
            self.refresh_preview()
        if self._full_preview is None:
            return
        popup=tk.Toplevel(self)
        popup.title("Prévia ampliada — figura completa")
        popup.geometry("1300x830")
        container=ttk.Frame(popup);container.pack(fill="both",expand=True)
        xscroll=ttk.Scrollbar(container,orient="horizontal")
        yscroll=ttk.Scrollbar(container,orient="vertical")
        canvas=tk.Canvas(container,bg="white",xscrollcommand=xscroll.set,
                         yscrollcommand=yscroll.set)
        xscroll.configure(command=canvas.xview)
        yscroll.configure(command=canvas.yview)
        xscroll.pack(side="bottom",fill="x")
        yscroll.pack(side="right",fill="y")
        canvas.pack(fill="both",expand=True)
        popup._image_ref=ImageTk.PhotoImage(self._full_preview)
        canvas.create_image(0,0,image=popup._image_ref,anchor="nw")
        canvas.configure(scrollregion=(0,0,*self._full_preview.size))

    def _settings(self):
        result={}
        for k,var in self._vars.items():
            result[k]=var.get().strip()
        for k in ("title_size","axis_size","tick_size",
                  "legend_size","legend_title_size"):
            try:
                size=float(result[k])
                if not 6<=size<=36:raise ValueError()
                result[k]=size
            except ValueError as exc:
                raise ValueError(f"Tamanho de fonte inválido para {k}: escolha de 6 a 36 pt.") from exc
        for k in ("background","title_color","axis_color","tick_color",
                  "legend_color","legend_title_color"):
            if not is_color_like(result[k]):
                raise ValueError(f"Cor inválida em {k}: use uma cor ou #RRGGBB.")
        # Largura/altura ajustáveis e verificadas antes da renderização.
        for key in ("figure_width","figure_height"):
            try:
                value=float(result[key].replace(",","."))
            except (ValueError,TypeError) as exc:
                raise ValueError("Dimensões inválidas: informe números em polegadas.") from exc
            if not 4 <= value <= 54:
                raise ValueError("A largura/altura devem estar entre 4 e 54 polegadas.")
            result[key]=value
        result["auto_expand"]=bool(self.auto_expand.get())
        result["background"]=result["background"].upper()
        aliases={}
        for name,var in self.alias_vars.items():
            new=var.get().strip()
            if not new:
                raise ValueError(f"Informe um nome para a classe {name}.")
            if new!=name:aliases[name]=new
        result["legend_names"]=aliases
        if not result["title_template"]:
            result["title_template"]="{intervalo}"
        return result

    def refresh_preview(self):
        self._preview_timer=None
        if not self.winfo_exists():
            return
        try:
            style=self._settings()
            panels=self.owner._display_panels()
            n=max(1,len(panels))
            self.figure.set_size_inches(style["figure_width"],style["figure_height"],
                                        forward=False)
            render_figure(self.figure,panels,self.owner.bounds,
                          self.owner._settings(),self.owner.palette,style=style)
            canvas_agg=FigureCanvasAgg(self.figure)
            canvas_agg.draw()
            self._full_preview=Image.frombuffer(
                "RGBA",canvas_agg.get_width_height(),
                bytes(canvas_agg.buffer_rgba()),"raw","RGBA",0,1)
            avail_w=max(400,self.preview.winfo_width()-10)
            avail_h=max(250,self.preview.winfo_height()-10)
            thumb=self._full_preview.copy()
            thumb.thumbnail((avail_w,avail_h),Image.Resampling.LANCZOS)
            self._photo=ImageTk.PhotoImage(thumb)
            self.preview.delete("all")
            self.preview.create_image(
                max(0,self.preview.winfo_width()//2),
                max(0,self.preview.winfo_height()//2),
                image=self._photo,anchor="center")
            final_w,final_h=self.figure.get_size_inches()
            self.status.set(
                f"Prévia/arquivo: {final_w:.1f} × {final_h:.1f} pol "
                f"({final_w*2.54:.1f} × {final_h*2.54:.1f} cm). "
                "Use 'Ampliar prévia' para conferir os textos.")
            return True
        except (ValueError,RuntimeError) as exc:
            self.status.set(str(exc))
            return False

    def _save(self):
        try:
            style=self._settings()
            # Atualiza a prévia com o mesmo estilo antes da escolha do destino.
            if not self.refresh_preview():
                raise ValueError(self.status.get())
            if self.gif:
                path=filedialog.asksaveasfilename(parent=self,title="Exportar GIF",
                        defaultextension=".gif",filetypes=[("GIF animado","*.gif")])
                if path:
                    self.owner._start_export_gif(path,style)
                    self.destroy()
            else:
                path=filedialog.asksaveasfilename(parent=self,title="Salvar figura",
                        defaultextension=".png",filetypes=[("PNG","*.png"),
                                                           ("SVG vetorial","*.svg"),
                                                           ("PDF","*.pdf")])
                if path:
                    self.owner._save_image_styled(path,style)
                    self.destroy()
        except (ValueError,RuntimeError) as exc:
            messagebox.showerror("Exportar figura",str(exc),parent=self)
