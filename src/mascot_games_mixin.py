#!/usr/bin/env python3
"""Mascot mini-games used by the ILT dashboard."""
from __future__ import annotations

import random
import tkinter as tk

try:
    from PIL import Image, ImageTk, ImageOps
except Exception:
    Image = ImageTk = ImageOps = None

from ui_resources import COLORS


class MascotGamesMixin:
    def _on_mascot_single_click(self, event=None):
        if self._mascot_click_job is not None:
            try:self.after_cancel(self._mascot_click_job)
            except Exception:pass
        self._mascot_click_job=self.after(220,self._run_mascot_single_click)

    def _run_mascot_single_click(self):
        self._mascot_click_job=None
        self._show_random_mascot()

    def _on_mascot_double_click(self, event=None):
        if self._mascot_click_job is not None:
            try:self.after_cancel(self._mascot_click_job)
            except Exception:pass
            self._mascot_click_job=None
        self._open_random_mascot_game()
        return 'break'

    def _open_random_mascot_game(self):
        """Randomly choose one mascot easter-egg game."""
        # If a game is already open, bring it forward instead of opening a
        # second game window on top of it.
        for win in (self._memory_win,self._hunt_win):
            try:
                if win is not None and win.winfo_exists():
                    win.deiconify(); win.lift(); win.focus_force(); return
            except Exception:
                pass
        # Randomness is deliberate. A small anti-repeat bias keeps both games
        # discoverable without making the sequence deterministic.
        choices=['memory','hunt']
        game=random.choice(choices)
        if game==self._last_mascot_game and random.random()<0.55:
            game='hunt' if game=='memory' else 'memory'
        self._last_mascot_game=game
        if game=='hunt':
            self._open_hunt_game()
        else:
            self._open_memory_game()

    def _show_random_mascot(self):
        if Image is None or ImageTk is None or not self.assets_dir.is_dir():
            if hasattr(self, 'mascot_box'):
                self.mascot_box.pack_forget()
            return
        files = sorted(self.assets_dir.glob('*.png'))
        if not files:
            self.mascot_box.pack_forget()
            return
        choices = [p for p in files if p != self._mascot_current] or files
        p = random.choice(choices)
        try:
            im = Image.open(p).convert('RGBA')
            im.thumbnail((138, 128), Image.Resampling.LANCZOS)
            self._mascot_photo = ImageTk.PhotoImage(im)
            self.mascot_label.configure(image=self._mascot_photo)
            self._mascot_current = p
        except Exception:
            pass

    def _open_memory_game(self):
        """Open a non-modal memory game without touching the running process."""
        if Image is None or ImageTk is None or not self.assets_dir.is_dir():
            return
        files=sorted(self.assets_dir.glob('*.png'))
        if not files:
            return
        if self._memory_win is not None:
            try:
                if self._memory_win.winfo_exists():
                    self._memory_win.deiconify(); self._memory_win.lift(); self._memory_win.focus_force()
                    return
            except Exception:
                self._memory_win=None

        win=tk.Toplevel(self)
        self._memory_win=win
        win.title(f'ILT — {self.tr("memory_game")}')
        win.configure(bg=COLORS['bg'])
        win.geometry('780x820')
        win.minsize(620,650)
        # Intentionally no grab_set(): classification/training continues and
        # the main dashboard remains usable while the game is open.
        top=tk.Frame(win,bg=COLORS['panel'])
        top.pack(fill='x')
        tk.Label(top,text=self.tr('memory_game'),bg=COLORS['panel'],fg=COLORS['text'],
                 font=('Sans',15,'bold')).pack(side='left',padx=16,pady=12)
        tk.Button(top,text=self.tr('memory_restart'),command=lambda:self._memory_start_level(0),
                  bg=COLORS['card2'],fg=COLORS['text'],activebackground=COLORS['border'],
                  activeforeground='white',relief='flat',bd=0,padx=12,pady=6,cursor='hand2').pack(side='right',padx=14)

        self._memory_status_var=tk.StringVar(value='')
        self._memory_message_var=tk.StringVar(value=self.tr('memory_hint'))
        tk.Label(win,textvariable=self._memory_status_var,bg=COLORS['bg'],fg=COLORS['cyan'],
                 font=('Monospace',10,'bold')).pack(pady=(10,2))
        tk.Label(win,textvariable=self._memory_message_var,bg=COLORS['bg'],fg=COLORS['muted'],
                 font=('Sans',9),wraplength=700,justify='center').pack(pady=(0,8))

        self._memory_board=tk.Frame(win,bg=COLORS['bg'])
        self._memory_board.pack(expand=True,padx=12,pady=8)
        self._memory_files=files
        self._memory_level_index=0
        self._memory_buttons=[]
        self._memory_cards=[]
        self._memory_images=[]
        self._memory_first=None
        self._memory_locked=False
        self._memory_attempts=0
        self._memory_errors=0
        self._memory_matched=set()
        self._memory_transition_job=None

        def close_game():
            if self._memory_transition_job is not None:
                try:self.after_cancel(self._memory_transition_job)
                except Exception:pass
                self._memory_transition_job=None
            try:win.destroy()
            except Exception:pass
            self._memory_win=None
        win.protocol('WM_DELETE_WINDOW',close_game)
        self._memory_start_level(0)

    def _memory_tinted_card(self,path,size,hue_shift=0):
        """Use the original image pixels; only shift hue to create more variants."""
        with Image.open(path) as src:
            rgba=src.convert('RGBA')
        alpha=rgba.getchannel('A')
        rgb=rgba.convert('RGB')
        if hue_shift and ImageOps is not None:
            hsv=rgb.convert('HSV')
            h,sat,val=hsv.split()
            shift=int(hue_shift)%256
            h=h.point(lambda px,sh=shift:(px+sh)%256)
            rgb=Image.merge('HSV',(h,sat,val)).convert('RGB')
        rgb.putalpha(alpha)
        rgb.thumbnail((size-8,size-8),Image.Resampling.LANCZOS)
        canvas=Image.new('RGBA',(size,size),(5,13,23,255))
        x=(size-rgb.width)//2; y=(size-rgb.height)//2
        canvas.alpha_composite(rgb,(x,y))
        return ImageTk.PhotoImage(canvas)

    def _memory_back_card(self,size):
        im=Image.new('RGB',(size,size),COLORS['card2'])
        try:
            from PIL import ImageDraw, ImageFont
            dr=ImageDraw.Draw(im)
            dr.rounded_rectangle((3,3,size-4,size-4),radius=max(8,size//10),
                                 outline=COLORS['blue'],width=2,fill=COLORS['card2'])
            font=None
            try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',max(22,size//3))
            except Exception:pass
            dr.text((size/2,size/2),'?',anchor='mm',fill=COLORS['muted'],font=font)
        except Exception:
            pass
        return ImageTk.PhotoImage(im)

    def _memory_start_level(self,level_index):
        if self._memory_win is None:
            return
        try:
            if not self._memory_win.winfo_exists():return
        except Exception:return
        if self._memory_transition_job is not None:
            try:self.after_cancel(self._memory_transition_job)
            except Exception:pass
            self._memory_transition_job=None

        self._memory_level_index=max(0,min(len(self._memory_levels)-1,int(level_index)))
        n=self._memory_levels[self._memory_level_index]
        total=n*n; pair_count=total//2; has_bonus=bool(total%2)
        self._memory_attempts=0; self._memory_errors=0; self._memory_first=None
        self._memory_locked=False; self._memory_matched=set(); self._memory_buttons=[]; self._memory_images=[]
        self._memory_message_var.set(self.tr('memory_hint'))
        for w in self._memory_board.winfo_children():
            w.destroy()

        size={3:132,4:112,5:92,6:76}.get(n,82)
        back=self._memory_back_card(size)
        self._memory_back_photo=back
        files=list(self._memory_files)
        random.shuffle(files)

        cards=[]
        hue_steps=[0,22,43,67,91,116,145,173,201,229]
        for pair_id in range(pair_count):
            path=files[pair_id % len(files)]
            cycle=pair_id//max(len(files),1)
            hue=random.choice(hue_steps) if cycle or pair_id>=len(files) else random.choice(hue_steps[:6])
            try:photo=self._memory_tinted_card(path,size,hue)
            except Exception:photo=self._memory_tinted_card(files[0],size,0)
            self._memory_images.append(photo)
            cards.extend([(pair_id,photo),(pair_id,photo)])
        random.shuffle(cards)

        if has_bonus:
            path=random.choice(files)
            hue=random.choice(hue_steps)
            bonus=self._memory_tinted_card(path,size,hue)
            self._memory_images.append(bonus)
            cards.insert(total//2,(None,bonus))
        self._memory_cards=cards

        for i,(pair_id,photo) in enumerate(cards):
            r,c=divmod(i,n)
            b=tk.Button(self._memory_board,image=back,command=lambda idx=i:self._memory_pick(idx),
                        bg=COLORS['card2'],activebackground=COLORS['border'],relief='flat',bd=0,
                        highlightthickness=1,highlightbackground=COLORS['border'],cursor='hand2')
            b.grid(row=r,column=c,padx=4,pady=4)
            self._memory_buttons.append(b)
            if pair_id is None:
                # Odd grids (3x3/5x5) cannot be made entirely from pairs. The
                # center card is a free bonus and does not affect attempts/errors.
                b.configure(image=photo,state='disabled',disabledforeground=COLORS['green'])
                self._memory_matched.add(i)
        self._memory_update_status()

    def _memory_update_status(self):
        n=self._memory_levels[self._memory_level_index]
        rate=(100.0*self._memory_errors/self._memory_attempts) if self._memory_attempts else 0.0
        self._memory_status_var.set(
            f'{self.tr("memory_level")}: {n}×{n}   •   '
            f'{self.tr("memory_attempts")}: {self._memory_attempts}   •   '
            f'{self.tr("memory_errors")}: {self._memory_errors}   •   '
            f'{self.tr("memory_error_rate")}: {rate:.1f}%'
        )

    def _memory_pick(self,index):
        if self._memory_locked or index in self._memory_matched:
            return
        if not (0<=index<len(self._memory_cards)):
            return
        pair_id,photo=self._memory_cards[index]
        if pair_id is None:
            return
        if self._memory_first is not None and self._memory_first==index:
            return
        try:self._memory_buttons[index].configure(image=photo)
        except Exception:return

        if self._memory_first is None:
            self._memory_first=index
            return

        first=self._memory_first; self._memory_first=None
        self._memory_attempts+=1
        first_pair,_=self._memory_cards[first]
        if first_pair==pair_id:
            self._memory_matched.update((first,index))
            self._memory_update_status()
            self._memory_check_complete()
        else:
            self._memory_errors+=1; self._memory_locked=True
            self._memory_update_status()
            self.after(620,lambda a=first,b=index:self._memory_hide_mismatch(a,b))

    def _memory_hide_mismatch(self,a,b):
        if self._memory_win is None:return
        for idx in (a,b):
            if idx in self._memory_matched:continue
            try:
                if self._memory_buttons[idx].winfo_exists():
                    self._memory_buttons[idx].configure(image=self._memory_back_photo)
            except Exception:pass
        self._memory_locked=False

    def _memory_check_complete(self):
        pair_positions=[i for i,(pid,_) in enumerate(self._memory_cards) if pid is not None]
        if not pair_positions or not all(i in self._memory_matched for i in pair_positions):
            return
        rate=(self._memory_errors/self._memory_attempts) if self._memory_attempts else 0.0
        if rate<0.30:
            if self._memory_level_index<len(self._memory_levels)-1:
                self._memory_message_var.set(self.tr('memory_advance'))
                nxt=self._memory_level_index+1
                self._memory_transition_job=self.after(1100,lambda:self._memory_start_level(nxt))
            else:
                self._memory_message_var.set(self.tr('memory_complete'))
        else:
            self._memory_message_var.set(self.tr('memory_repeat'))
            cur=self._memory_level_index
            self._memory_transition_job=self.after(1200,lambda:self._memory_start_level(cur))

    def _open_hunt_game(self):
        """Open a dense mascot-search game inspired by visual search puzzles."""
        if Image is None or ImageTk is None or not self.assets_dir.is_dir():
            return
        files=sorted(self.assets_dir.glob('*.png'))
        if not files:
            return
        if self._hunt_win is not None:
            try:
                if self._hunt_win.winfo_exists():
                    self._hunt_win.deiconify(); self._hunt_win.lift(); self._hunt_win.focus_force()
                    return
            except Exception:
                self._hunt_win=None

        win=tk.Toplevel(self)
        self._hunt_win=win
        win.title(f'ILT — {self.tr("hunt_game")}')
        win.configure(bg=COLORS['bg'])
        win.geometry('980x760')
        win.minsize(760,620)

        top=tk.Frame(win,bg=COLORS['panel'])
        top.pack(fill='x')
        tk.Label(top,text=self.tr('hunt_game'),bg=COLORS['panel'],fg=COLORS['text'],
                 font=('Sans',15,'bold')).pack(side='left',padx=16,pady=12)
        tk.Button(top,text=self.tr('hunt_new_round'),command=self._hunt_new_round,
                  bg=COLORS['blue2'],fg='white',activebackground=COLORS['blue'],
                  activeforeground='white',relief='flat',bd=0,padx=12,pady=6,
                  cursor='hand2').pack(side='right',padx=14)

        info=tk.Frame(win,bg=COLORS['bg'])
        info.pack(fill='x',padx=18,pady=(12,6))
        target_box=tk.Frame(info,bg=COLORS['card2'],highlightbackground=COLORS['border'],highlightthickness=1)
        target_box.pack(side='left')
        tk.Label(target_box,text=self.tr('hunt_target'),bg=COLORS['card2'],fg=COLORS['muted'],
                 font=('Sans',8,'bold')).pack(anchor='w',padx=8,pady=(5,1))
        self._hunt_target_label=tk.Label(target_box,bg=COLORS['log_bg'],width=88,height=72)
        self._hunt_target_label.pack(padx=8,pady=(0,7))

        text_box=tk.Frame(info,bg=COLORS['bg'])
        text_box.pack(side='left',fill='x',expand=True,padx=(14,0))
        self._hunt_status_var=tk.StringVar(value=self.tr('hunt_hint'))
        self._hunt_attempts_var=tk.StringVar(value=f'{self.tr("hunt_attempts")}: 0')
        tk.Label(text_box,textvariable=self._hunt_attempts_var,bg=COLORS['bg'],fg=COLORS['cyan'],
                 font=('Monospace',10,'bold')).pack(anchor='w')
        tk.Label(text_box,textvariable=self._hunt_status_var,bg=COLORS['bg'],fg=COLORS['muted'],
                 font=('Sans',9),wraplength=660,justify='left').pack(anchor='w',pady=(5,0))

        board_frame=tk.Frame(win,bg=COLORS['card'],highlightbackground=COLORS['border'],highlightthickness=1)
        board_frame.pack(fill='both',expand=True,padx=18,pady=(6,18))
        self._hunt_canvas=tk.Canvas(board_frame,bg=COLORS['card'],highlightthickness=0,cursor='crosshair')
        self._hunt_canvas.pack(fill='both',expand=True,padx=4,pady=4)

        self._hunt_files=files
        self._hunt_photos=[]
        self._hunt_target_photo=None
        self._hunt_target_item=None
        self._hunt_attempts=0
        self._hunt_locked=False

        def close_game():
            if self._hunt_round_job is not None:
                try:self.after_cancel(self._hunt_round_job)
                except Exception:pass
                self._hunt_round_job=None
            try:win.destroy()
            except Exception:pass
            self._hunt_win=None
        win.protocol('WM_DELETE_WINDOW',close_game)
        self.after(80,self._hunt_new_round)

    def _hunt_variant_photo(self,path,max_size,hue_shift=0,angle=0,mirror=False):
        """Create one visually varied mascot while preserving the source artwork."""
        with Image.open(path) as src:
            rgba=src.convert('RGBA')
        alpha=rgba.getchannel('A')
        rgb=rgba.convert('RGB')
        if hue_shift:
            hsv=rgb.convert('HSV')
            h,sat,val=hsv.split()
            shift=int(hue_shift)%256
            h=h.point(lambda px,sh=shift:(px+sh)%256)
            rgb=Image.merge('HSV',(h,sat,val)).convert('RGB')
        rgb.putalpha(alpha)
        rgba=rgb
        if mirror and ImageOps is not None:
            rgba=ImageOps.mirror(rgba)
        rgba.thumbnail((max_size,max_size),Image.Resampling.LANCZOS)
        if angle:
            rgba=rgba.rotate(float(angle),resample=Image.Resampling.BICUBIC,expand=True)
            rgba.thumbnail((max_size,max_size),Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(rgba)

    def _hunt_target_preview(self,path):
        with Image.open(path) as src:
            im=src.convert('RGBA')
        im.thumbnail((82,68),Image.Resampling.LANCZOS)
        canvas=Image.new('RGBA',(88,72),(5,13,23,255))
        canvas.alpha_composite(im,((88-im.width)//2,(72-im.height)//2))
        return ImageTk.PhotoImage(canvas)

    def _hunt_new_round(self):
        if self._hunt_win is None:
            return
        try:
            if not self._hunt_win.winfo_exists():return
        except Exception:return
        if self._hunt_round_job is not None:
            try:self.after_cancel(self._hunt_round_job)
            except Exception:pass
            self._hunt_round_job=None

        files=list(getattr(self,'_hunt_files',[]))
        if not files:
            return
        canvas=self._hunt_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        width=max(int(canvas.winfo_width()),720)
        height=max(int(canvas.winfo_height()),430)
        self._hunt_attempts=0
        self._hunt_locked=False
        self._hunt_target_item=None
        self._hunt_status_var.set(self.tr('hunt_hint'))
        self._hunt_attempts_var.set(f'{self.tr("hunt_attempts")}: 0')
        self._hunt_photos=[]

        target_path=random.choice(files)
        try:
            self._hunt_target_photo=self._hunt_target_preview(target_path)
            self._hunt_target_label.configure(image=self._hunt_target_photo)
            self._hunt_target_label.image=self._hunt_target_photo
        except Exception:
            self._hunt_target_photo=None

        # A dense but non-overlapping jittered grid gives the visual-search feel
        # of a "find the character" puzzle while keeping every click unambiguous.
        cols=10 if width>=850 else 8
        rows=6 if height>=500 else 5
        total=cols*rows
        target_index=random.randrange(total)
        cell_w=width/cols
        cell_h=height/rows

        # Background clutter helps the target blend in without using any external
        # artwork or copyrighted characters.
        clutter_colors=[COLORS['card2'],COLORS['border'],'#24242a','#1f1f23','#26262c']
        for _ in range(max(24,total//2)):
            x=random.randint(8,max(9,width-8)); y=random.randint(8,max(9,height-8))
            r=random.randint(5,18)
            col=random.choice(clutter_colors)
            if random.random()<0.5:
                canvas.create_oval(x-r,y-r,x+r,y+r,outline=col,width=1)
            else:
                canvas.create_rectangle(x-r,y-r,x+r,y+r,outline=col,width=1)

        hue_steps=[18,36,58,82,104,128,151,176,201,225]
        for idx in range(total):
            row,col=divmod(idx,cols)
            cx=(col+0.5)*cell_w+random.uniform(-cell_w*0.18,cell_w*0.18)
            cy=(row+0.5)*cell_h+random.uniform(-cell_h*0.16,cell_h*0.16)
            size=int(max(42,min(68,min(cell_w,cell_h)*0.78)))
            is_target=idx==target_index

            if is_target:
                path=target_path; hue=0; angle=0; mirror=False
            else:
                path=random.choice(files)
                # If the same source appears as a decoy, always transform it so
                # there is exactly one exact target in the crowd.
                same_source=(path==target_path)
                hue=random.choice(hue_steps) if same_source or random.random()<0.68 else 0
                angle=random.choice([-24,-18,-12,-7,7,12,18,24]) if same_source or random.random()<0.72 else 0
                mirror=bool(random.getrandbits(1)) if same_source or random.random()<0.45 else False
            try:
                photo=self._hunt_variant_photo(path,size,hue,angle,mirror)
            except Exception:
                continue
            self._hunt_photos.append(photo)
            item=canvas.create_image(cx,cy,image=photo,anchor='center')
            if is_target:
                self._hunt_target_item=item
            canvas.tag_bind(item,'<Button-1>',lambda e,item=item:self._hunt_pick(item))

        # Ensure the target is always a valid item even if an image unexpectedly
        # failed to load.
        if self._hunt_target_item is None:
            self._hunt_status_var.set(
                'Não foi possível montar esta rodada.' if self.lang=='pt'
                else 'Could not build this round.'
            )

    def _hunt_pick(self,item):
        if self._hunt_locked:
            return
        self._hunt_attempts+=1
        self._hunt_attempts_var.set(f'{self.tr("hunt_attempts")}: {self._hunt_attempts}')
        if item==self._hunt_target_item:
            self._hunt_locked=True
            self._hunt_status_var.set(self.tr('hunt_found'))
            try:self._hunt_canvas.itemconfigure(item,state='normal')
            except Exception:pass
            self._hunt_round_job=self.after(950,self._hunt_new_round)
        else:
            self._hunt_status_var.set(self.tr('hunt_miss'))
