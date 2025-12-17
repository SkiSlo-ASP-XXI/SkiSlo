import xml.etree.ElementTree as ET
import numpy as np
import laspy
import argparse
from pathlib import Path
from datetime import datetime


def parse_landxml(xml_file:str)->tuple[dict, dict]:
    """
    Parsa il file XML LandXML ed estrae i punti e le loro proprietà.
    
    Args:
        xml_file: Percorso del file XML
        
    Returns:
        tuple: (points_dict, features_dict)
            - points_dict: Dizionario {name: (x, y, z)}
            - features_dict: Dizionario {name: {property: value}}
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    ns = {'l': 'http://www.landxml.org/schema/LandXML-1.2'}
    
    points_dict = {}
    features_dict = {}
    
    cgpoints = root.findall('.//l:CgPoint', ns) or root.findall('.//CgPoint')
    
    for cgpoint in cgpoints:
        name = cgpoint.get('name', None)
        if name is None: continue
            
        coords_text = cgpoint.text
        if coords_text is None: continue
        
        try:
            x, y, z = list(map(float, coords_text.strip().split()))
            points_dict[name] = (x, y, z)
        except (ValueError, AttributeError):
            print(f"Avvertenza: Impossibile parsare le coordinate per il punto {name}")
            continue
    
    # Estrai i Feature properties - prova con namespace e senza
    features = root.findall('.//l:Feature', ns) or root.findall('.//Feature')
    
    for feature in features:
        feat_name = feature.get('name', None)
        if feat_name is None: continue
            
        properties = {}
        props = feature.findall('l:Property', ns) or feature.findall('Property')
        for prop in props:
            label = prop.get('label', None)
            value = prop.get('value', None)
            if label and value:
                properties[label] = value
        
        features_dict[feat_name] = properties
    
    return points_dict, features_dict


def create_las_file(output_file, points_dict, features_dict):
    """
    Crea un file LAS dai dati estratti dall'XML.
    
    Args:
        output_file: Percorso del file LAS di output
        points_dict: Dizionario {name: (x, y, z)}
        features_dict: Dizionario {name: {property: value}}
    """
    
    if not points_dict:
        print("Errore: Nessun punto trovato nel file XML")
        return False
    
    # Estrai le coordinate
    names = list(points_dict.keys())
    coords = np.array([points_dict[name] for name in names])
    
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    las = laspy.create()
    
    las.x, las.y, las.z = y, x, z
    
    # Check una sola volta per i dati opzionali
    has_pass_count = any('PassCount' in features_dict.get(name, {}) for name in names)
    has_delta_h = any('deltaH' in features_dict.get(name, {}) for name in names)
    
    # Inizializza gli array
    intensity = np.zeros(len(names), dtype=np.uint16)
    pass_count = np.zeros(len(names), dtype=np.uint8) if has_pass_count else None
    delta_h = np.zeros(len(names), dtype=np.float32) if has_delta_h else None
    
    # Unico ciclo per tutti i dati
    for i, name in enumerate(names):
        if name in features_dict:
            # Processo Speed
            if 'Speed' in features_dict[name]:
                try:
                    speed = float(features_dict[name]['Speed'])
                    # Scala la velocità a intensità (0-65535)
                    intensity[i] = int(min(speed * 100, 65535))
                except (ValueError, KeyError):
                    pass
                
            if pass_count is not None and 'PassCount' in features_dict[name]:
                try:
                    pass_count[i] = int(features_dict[name]['PassCount'])
                except ValueError:
                    pass
            
            # Processo deltaH
            if delta_h is not None and 'deltaH' in features_dict[name]:
                try:
                    delta_h[i] = float(features_dict[name]['deltaH'])
                except ValueError:
                    pass
    
    las.intensity = intensity
    
    if pass_count is not None:
        try:
            las.PassCount = pass_count
        except Exception as e:
            print(f"Avvertenza: Impossibile aggiungere PassCount: {e}")
    
    if delta_h is not None:
        try:
            las.deltaH = delta_h
        except Exception as e:
            print(f"Avvertenza: Impossibile aggiungere deltaH: {e}")
    
    # Salva il file
    las.write(output_file)
    print(f"File LAS creato con successo: {output_file}")
    print(f"Numero di punti: {len(names)}")
    print(f"Numero di feature: {len(features_dict)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convertire file XML (LandXML) a formato LAS (LIDAR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Esempi di utilizzo:
            python xml_to_las_converter.py Sestriere_DTM_cgpoints_.xml
            python xml_to_las_converter.py input.xml output.las
        """
    )
    
    parser.add_argument('input_file', help='Percorso del file XML di input')
    parser.add_argument('output_file', nargs='?', help='Percorso del file LAS di output (opzionale)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    # Controlla se il file di input esiste
    if not input_path.exists():
        print(f"Errore: Il file '{input_path}' non esiste")
        return False
    
    # Definisci il file di output
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix('.las')
    
    print(f"Conversione in corso...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Parsa il file XML
    print("\nParsing del file XML...")
    points_dict, features_dict = parse_landxml(input_path)
    print(f"Punti trovati: {len(points_dict)}")
    print(f"Feature trovate: {len(features_dict)}")
    
    # Crea il file LAS
    print("\nCreazione del file LAS...")
    return  create_las_file(output_path, points_dict, features_dict)
    

if __name__ == '__main__':
    assert main(), "Qualcosa è andato storto!"
