import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
# Importamos el procesador desde el nuevo archivo modular
from src.processor import RunningProcessor 

class VentanaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3D Running Analysis Studio v1.0")
        self.geometry("600x750")
        ctk.set_appearance_mode("dark")
        
        # Variables de ruta y parámetros
        self.rutas = {
            'mdx': ctk.StringVar(),
            'standing': ctk.StringVar(),
            'running': ctk.StringVar()
        }
        self.velocidad = ctk.StringVar(value="12.0")
        self.long_pierna = ctk.StringVar(value="900")

        self._build_ui()

    def _build_ui(self):
        """Organiza los elementos de la interfaz"""
        ctk.CTkLabel(self, text="CARGAR ARCHIVOS DE ESTUDIO", font=("Arial", 18, "bold")).pack(pady=20)
        
        self.crear_input("Archivo de Sesión (MDX):", self.rutas['mdx'])
        self.crear_input("Archivo Standing (TDF):", self.rutas['standing'])
        self.crear_input("Archivo Running (TDF):", self.rutas['running'])

        ctk.CTkLabel(self, text="PARÁMETROS DEL ANÁLISIS", font=("Arial", 15, "bold")).pack(pady=(20, 10))
        
        # Frame de parámetros numéricos
        frame_params = ctk.CTkFrame(self, fg_color="transparent")
        frame_params.pack(fill="x", padx=40, pady=5)

        self._crear_campo_numerico(frame_params, "Velocidad (km/h):", self.velocidad, side="left")
        self._crear_campo_numerico(frame_params, "Long. Pierna (mm):", self.long_pierna, side="right")

        # Estado y Barra de progreso
        self.lbl_status = ctk.CTkLabel(self, text="Esperando inicio...", font=("Arial", 12, "italic"), text_color="gray")
        self.lbl_status.pack(pady=(30, 0))
        
        self.pbar = ctk.CTkProgressBar(self, width=450)
        self.pbar.pack(pady=10)
        self.pbar.set(0)

        # Botones de acción
        self.btn_run = ctk.CTkButton(self, text="INICIAR ANÁLISIS", command=self.lanzar_hilo, 
                                     height=50, width=300, font=("Arial", 14, "bold"))
        self.btn_run.pack(pady=(20, 10))

        ctk.CTkButton(self, text="SALIR", command=self.confirmar_salida, fg_color="#A93226").pack(pady=10)

    def crear_input(self, texto, var):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=40, pady=5)
        ctk.CTkLabel(f, text=texto).pack(side="top", anchor="w")
        ctk.CTkEntry(f, textvariable=var).pack(side="left", fill="x", expand=True, pady=5)
        ctk.CTkButton(f, text="...", width=40, command=lambda: self.seleccionar(var)).pack(side="right", padx=(5,0))

    def _crear_campo_numerico(self, parent, label, var, side):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(side=side, expand=True, fill="x", padx=10)
        ctk.CTkLabel(f, text=label, font=("Arial", 11)).pack(anchor="w")
        ctk.CTkEntry(f, textvariable=var).pack(fill="x", pady=5)

    def seleccionar(self, var):
        p = filedialog.askopenfilename()
        if p: var.set(p)

    def actualizar_ui(self, val, texto):
        """Callback que será llamado desde el procesador para actualizar la barra"""
        self.pbar.set(val)
        self.lbl_status.configure(text=texto)
        self.update_idletasks()

    def lanzar_hilo(self):
        if not self.rutas['standing'].get() or not self.rutas['running'].get():
            messagebox.showwarning("Atención", "Selecciona los archivos TDF.")
            return
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        """Lógica de segundo plano que conecta la UI con el Procesador"""
        try:
            vel_val = float(self.velocidad.get())
            long_val = float(self.long_pierna.get())
            rutas_puras = {k: v.get() for k, v in self.rutas.items()}
            
            # Instanciamos el procesador modular
            proc = RunningProcessor(rutas_puras, self.actualizar_ui, velocidad=vel_val, longitud=long_val)
            exito, msg = proc.ejecutar()
            
            self.after(0, lambda: self.finalizar(exito, msg))
        except ValueError:
            self.after(0, lambda: self.finalizar(False, "Ingresa valores numéricos válidos."))

    def finalizar(self, exito, msg):
        self.btn_run.configure(state="normal")
        if exito:
            self.pbar.configure(progress_color="#27ae60")
            messagebox.showinfo("Completado", "Análisis finalizado con éxito.")
        else:
            self.pbar.configure(progress_color="#a93226")
            messagebox.showerror("Error", f"Ocurrió un problema: {msg}")

    def confirmar_salida(self):
        if messagebox.askyesno("Salir", "¿Desea cerrar la aplicación?"):
            self.destroy()

if __name__ == "__main__":
    VentanaApp().mainloop() 