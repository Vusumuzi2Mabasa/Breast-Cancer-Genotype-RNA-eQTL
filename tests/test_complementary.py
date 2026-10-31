import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from complementary import capture,reconstruction_r2


class ProgrammeDiagnostics(unittest.TestCase):
    def test_programme_capture_rotation_and_missing_dimension(self):
        rng=np.random.default_rng(401);h=rng.normal(size=(90,3))
        q,_=np.linalg.qr(h-h.mean(0));rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        self.assertAlmostEqual(capture(q@rotation,h),1,12)
        self.assertAlmostEqual(capture(q[:,:1],h),1/3,12)

    def test_unrepresented_programmes(self):
        rng=np.random.default_rng(402);a=rng.normal(size=(90,6));q,_=np.linalg.qr(a-a.mean(0))
        self.assertAlmostEqual(capture(q[:,:3],q[:,3:]),0,12)

    def test_projection_reconstruction_reference(self):
        rng=np.random.default_rng(403);z=rng.normal(size=(80,2));z-=z.mean(0);w=rng.normal(size=(5,2));x=z@w.T
        self.assertAlmostEqual(reconstruction_r2(x,z,w),1,12)
        self.assertAlmostEqual(reconstruction_r2(x,np.zeros_like(z),w),0,12)


if __name__=='__main__':unittest.main()
