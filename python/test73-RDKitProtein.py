import glob, os
import math, gzip
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem.rdchem import Atom

RDLogger.DisableLog('rdApp.*')  




def rot_ar_x(radi):
    return  np.array([[1, 0, 0, 0],
                      [0, np.cos(radi), -np.sin(radi), 0],
                      [0, np.sin(radi), np.cos(radi), 0],
                     [0, 0, 0, 1]], dtype=np.double)
 
def rot_ar_y(radi):
    return  np.array([[np.cos(radi), 0, np.sin(radi), 0],
                      [0, 1, 0, 0],
                      [-np.sin(radi), 0, np.cos(radi), 0],
                     [0, 0, 0, 1]], dtype=np.double)
 
def rot_ar_z(radi):
    return  np.array([[np.cos(radi), -np.sin(radi), 0, 0],
                      [np.sin(radi), np.cos(radi), 0, 0],
                      [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.double)

def move_xyz(dxyz):
    return np.array( [
        [ 1, 0, 0, dxyz[0] ],
        [ 0, 1, 0, dxyz[1] ],
        [ 0, 0, 1, dxyz[2] ],
        [ 0, 0, 0, 1 ] ], dtype=np.double )

transforms = { 0: rot_ar_x, 1: rot_ar_y, 2: rot_ar_z, 3:move_xyz }



def rotate_molecule( mole, dx, dy ) :
    m = Chem.Mol( mol )
    conf = m.GetConformer(0)
    center = Chem.rdMolTransforms.ComputeCentroid( conf  )
    Chem.rdMolTransforms.TransformConformer (conf, transforms[3]([-center.x, -center.y, -center.z]) )
    Chem.rdMolTransforms.TransformConformer (conf, transforms[0](dx) )
    Chem.rdMolTransforms.TransformConformer (conf, transforms[1](dy) )
    Chem.rdMolTransforms.TransformConformer (conf, transforms[3]([center.x, center.y, center.z]) )
    return m



def rotate_molecule_to_files( mol, numrot, folder ) :
    for ix in range(numrot) :
        for iy in range(numrot) :
            dx = ix*2*np.pi/numrot
            dy = iy*2*np.pi/numrot
            m = rotate_molecule( mol, dx, dy )

            fname = folder + 'ligand-{0:0>2}{1:0>2}.pdb'.format(ix,iy)
            Chem.rdmolfiles.MolToPDBFile( m, fname )




def get_xyz_CA_protein( prot ) :
    xyz = []
    conf = prot.GetConformer(0)
    for atm in prot.GetAtoms():
        name = atm.GetPDBResidueInfo().GetName().strip()
        if 'CA' == name :
            idx = atm.GetIdx()
            x = conf.GetAtomPosition(idx)
            xyz.append( x )
    return xyz



def calc_rmsd( xyza, xyzb ) :
    num = len( xyza )
    for i in range(num) :
        xa = xyza[i]
        xb = xyzb[i]
        d2 = (xa[0] - xb[0])*(xa[0] - xb[0] ) + (xa[1] - xb[1])*(xa[1] - xb[1] ) + (xa[2] - xb[2])*(xa[2] - xb[2] )
    return math.sqrt(d2/num)



def load_pdb_in_directory( dire ) :
    moles = []
    for fname in glob.glob( dire ) :
        m = Chem.rdmolfiles.MolFromPDBFile( fname )
        if m : 
            moles.append(m)
    return moles


def load_mae( fname ) :
    with Chem.rdmolfiles.MaeMolSupplier( gzip.open( fname ) ) as suppl :
        moles = [m for m in suppl if m ]
        return moles
    return None



prot = Chem.rdmolfiles.MolFromPDBFile( "../data/ligand.pdb" )

if False:
    rotate_molecule_to_files( prot, folder='../tmp/', numrot = 12 )

    for ckey, chain in Chem.SplitMolByPDBChainId(prot).items():
        print( ckey, chain )
        for rkey, res in Chem.SplitMolByPDBResidues(chain).items():
            print( rkey, res )

    for rkey, res in Chem.SplitMolByPDBResidues(prot).items():
        print( rkey, res )

    chains = {a.GetPDBResidueInfo().GetChainId() for a in prot.GetAtoms()}
    for c in chains:
        print(c)

    xyzs = get_xyz_CA_protein( prot )
    print( xyzs )



if True:
    # moles = load_pdb_in_directory( dire="../dock/*.pdb" )
    moles = load_mae( fname="../data/dock.maegz" )

    for m in moles :
        m.SetProp( 'center', 'Y' )

    nmoles = len( moles )
    for i in range(nmoles) :
        if 'Y' != moles[i].GetProp( 'center' ) :
            continue
        xyzi = get_xyz_CA_protein( moles[i] )
        for j in range( i+1, nmoles ) :
            if 'Y' != moles[j].GetProp( 'center' ) :
                continue
            xyzj = get_xyz_CA_protein( moles[j] )
            rmsd = calc_rmsd(xyzi, xyzj )
            if rmsd < 2.0 :
                moles[j].SetProp( 'center', 'N' )
                # print( 'merged : ', i, j, rmsd  )


    for i in range(nmoles) :
        if 'Y' == moles[i].GetProp( 'center' ) :
            print( 'center : ', i )



