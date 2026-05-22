import xml.etree.ElementTree as ET

def write_vtm(vtm_path, block_files):
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    mbds = ET.SubElement(root, "vtkMultiBlockDataSet")
    for i, (name, file_path) in enumerate(block_files.items()):
        ET.SubElement(mbds, "DataSet", index=str(i), file=file_path, name=name)
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(vtm_path, encoding="utf-8", xml_declaration=True)

write_vtm("examples/output_2d_flapping/wing_skin_sim_combined.vtm", {
    "Skin": "wing_skin_sim_skin.vtu",
    "Beams": "wing_skin_sim_beams.vtp"
})
