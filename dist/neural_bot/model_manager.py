"""
Model Manager - Gestión de modelos neurales con nombres personalizados

Permite guardar, cargar y gestionar múltiples modelos entrenados con nombres
descriptivos en lugar de solo versionado numérico.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import joblib


class ModelManager:
    """Gestor de modelos neurales con sistema de nombrado flexible"""
    
    def __init__(self, models_dir='models'):
        """
        Inicializa el gestor de modelos
        
        Args:
            models_dir: Directorio raíz para almacenar modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.models_dir / 'models_index.json'
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Carga el índice de modelos"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error cargando índice: {e}")
                return {'default_model': None, 'models': {}}
        return {'default_model': None, 'models': {}}
    
    def _save_index(self):
        """Guarda el índice de modelos"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error guardando índice: {e}")
    
    def save_model(self, model, scaler, name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Guarda un modelo con nombre personalizado
        
        Args:
            model: Modelo de Keras entrenado
            scaler: Scaler de sklearn
            name: Nombre descriptivo del modelo (ej: 'eth_optimized')
            metadata: Metadata adicional (símbolos, métricas, etc.)
        
        Returns:
            bool: True si se guardó correctamente
        """
        # Validar nombre
        if not name or not name.replace('_', '').replace('-', '').isalnum():
            print(f"❌ Nombre inválido: {name}")
            return False
        
        # Crear directorio del modelo
        model_path = self.models_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Guardar modelo de Keras
            model_file = model_path / 'model.keras'
            model.save(model_file)
            print(f"💾 Modelo guardado: {model_file}")
            
            # Guardar scaler
            scaler_file = model_path / 'scaler.pkl'
            joblib.dump(scaler, scaler_file)
            print(f"💾 Scaler guardado: {scaler_file}")
            
            # Preparar metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'name': name,
                'saved_at': datetime.now().isoformat(),
                'model_file': str(model_file),
                'scaler_file': str(scaler_file),
            })
            
            # Guardar metadata
            metadata_file = model_path / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"💾 Metadata guardada: {metadata_file}")
            
            # Actualizar índice
            self.index['models'][name] = {
                'path': str(model_path),
                'created_at': metadata['saved_at'],
                'metadata_summary': {
                    'symbols': metadata.get('symbols', []),
                    'accuracy': metadata.get('accuracy', None),
                    'timeframe': metadata.get('timeframe', None),
                }
            }
            
            # Si es el primer modelo, marcarlo como default
            if self.index['default_model'] is None:
                self.index['default_model'] = name
                print(f"✅ '{name}' marcado como modelo por defecto")
            
            self._save_index()
            print(f"✅ Modelo '{name}' guardado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            # Limpiar en caso de error
            if model_path.exists():
                shutil.rmtree(model_path)
            return False
    
    def load_model(self, name: Optional[str] = None) -> Optional[tuple]:
        """
        Carga un modelo por nombre
        
        Args:
            name: Nombre del modelo. Si es None, carga el default
        
        Returns:
            tuple(model, scaler, metadata) o None si falla
        """
        # Si no se especifica nombre, usar default
        if name is None:
            name = self.index.get('default_model')
            if name is None:
                print("❌ No hay modelo por defecto configurado")
                return None
            print(f"📂 Cargando modelo por defecto: {name}")
        
        # Verificar que existe
        if name not in self.index['models']:
            print(f"❌ Modelo '{name}' no encontrado")
            return None
        
        model_path = Path(self.index['models'][name]['path'])
        
        if not model_path.exists():
            print(f"❌ Directorio del modelo no existe: {model_path}")
            return None
        
        try:
            # Cargar modelo
            import tensorflow as tf
            model_file = model_path / 'model.keras'
            # safe_mode=False permite cargar modelos con layers Lambda (ej: attention)
            model = tf.keras.models.load_model(model_file, safe_mode=False)
            print(f"📂 Modelo cargado: {model_file}")
            
            # Cargar scaler
            scaler_file = model_path / 'scaler.pkl'
            scaler = joblib.load(scaler_file)
            print(f"📂 Scaler cargado: {scaler_file}")
            
            # Cargar metadata
            metadata_file = model_path / 'metadata.json'
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"📂 Metadata cargada: {metadata_file}")
            
            return model, scaler, metadata
            
        except Exception as e:
            print(f"❌ Error cargando modelo '{name}': {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos los modelos disponibles con su metadata
        
        Returns:
            List de dicts con información de cada modelo
        """
        models_list = []
        
        for name, info in self.index['models'].items():
            model_info = {
                'name': name,
                'is_default': (name == self.index['default_model']),
                'created_at': info['created_at'],
                'path': info['path'],
            }
            model_info.update(info.get('metadata_summary', {}))
            models_list.append(model_info)
        
        # Ordenar por fecha de creación (más reciente primero)
        models_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models_list
    
    def set_default_model(self, name: str) -> bool:
        """
        Marca un modelo como predeterminado
        
        Args:
            name: Nombre del modelo
        
        Returns:
            bool: True si se estableció correctamente
        """
        if name not in self.index['models']:
            print(f"❌ Modelo '{name}' no encontrado")
            return False
        
        old_default = self.index['default_model']
        self.index['default_model'] = name
        self._save_index()
        
        print(f"✅ Modelo por defecto cambiado: {old_default} → {name}")
        return True
    
    def delete_model(self, name: str, force: bool = False) -> bool:
        """
        Elimina un modelo
        
        Args:
            name: Nombre del modelo
            force: Si True, permite eliminar el modelo default
        
        Returns:
            bool: True si se eliminó correctamente
        """
        if name not in self.index['models']:
            print(f"❌ Modelo '{name}' no encontrado")
            return False
        
        # Proteger modelo default
        if name == self.index['default_model'] and not force:
            print(f"⚠️ '{name}' es el modelo por defecto. Usa force=True para eliminarlo")
            return False
        
        try:
            # Eliminar directorio
            model_path = Path(self.index['models'][name]['path'])
            if model_path.exists():
                shutil.rmtree(model_path)
                print(f"🗑️ Directorio eliminado: {model_path}")
            
            # Actualizar índice
            del self.index['models'][name]
            
            # Si era el default, limpiar
            if name == self.index['default_model']:
                # Intentar asignar otro modelo como default
                if self.index['models']:
                    new_default = list(self.index['models'].keys())[0]
                    self.index['default_model'] = new_default
                    print(f"✅ Nuevo modelo por defecto: {new_default}")
                else:
                    self.index['default_model'] = None
            
            self._save_index()
            print(f"✅ Modelo '{name}' eliminado")
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando modelo: {e}")
            return False
    
    def get_model_metadata(self, name: str) -> Optional[Dict]:
        """
        Obtiene la metadata completa de un modelo
        
        Args:
            name: Nombre del modelo
        
        Returns:
            Dict con metadata o None
        """
        if name not in self.index['models']:
            print(f"❌ Modelo '{name}' no encontrado")
            return None
        
        model_path = Path(self.index['models'][name]['path'])
        metadata_file = model_path / 'metadata.json'
        
        if not metadata_file.exists():
            print(f"⚠️ Archivo de metadata no encontrado para '{name}'")
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error leyendo metadata: {e}")
            return None
    
    def get_default_model_name(self) -> Optional[str]:
        """Retorna el nombre del modelo por defecto"""
        return self.index.get('default_model')
