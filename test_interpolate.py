from skymap import SkyMapUtils as smu
import matplotlib.pyplot as plt

data_ring, h, event, data_nest = smu.read_a_skymap()
smu.visualize(data_ring, title=None)
h = dict(h)
pmap = smu.interpolate_sky_map(data_nest, h['NSIDE'], image=True)

