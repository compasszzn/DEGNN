from synthetic_sim import ChargedParticlesSim, SpringSim, GravitySim
import time
import numpy as np
import argparse

"""
nbody_small:   python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix charged
gravity_small: python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small
"""

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=200,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_observed', type=int, default=200,
                    help='Number of balls in the simulation.')
parser.add_argument('--n_unobserved', type=int, default=0,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_observed + args.n_unobserved)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_observed + args.n_unobserved, vel_norm=initial_vel_norm)
    suffix = '_charged'
elif args.simulation == 'gravity':
    sim = GravitySim(noise_var=0.0, n_balls=args.n_observed + args.n_unobserved, vel_norm=initial_vel_norm)
    suffix = '_gravity'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_observed) + str(args.n_unobserved) + "_initvel%d" % args.initial_vel + args.suffix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq,dataset):
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()
    force_save_all = list()
    mass_all = list()
    for i in range(num_sims):
        t = time.time()
        if dataset=='charged':
            loc, vel, edges, charges = sim.sample_trajectory(T=length,
                                                            sample_freq=sample_freq)
            loc = loc[:, :, :args.n_observed]
            vel = vel[:, :, :args.n_observed]
            edges = edges[:args.n_observed, :args.n_observed]
            charges = charges[:args.n_observed]

            loc_all.append(loc)
            vel_all.append(vel)
            edges_all.append(edges)
            charges_all.append(charges)
        elif dataset=='gravity':
            loc, vel, force_save, mass = sim.sample_trajectory(T=length,
                                                            sample_freq=sample_freq)
            loc = loc[:, :, :args.n_observed]
            vel = vel[:, :, :args.n_observed]
            force_save = force_save[:, :, :args.n_observed]
            mass = mass[:args.n_observed]

            loc_all.append(loc)
            vel_all.append(vel)
            force_save_all.append(force_save)
            mass_all.append(mass)
        elif dataset=='springs':
            loc, vel, edges = sim.sample_trajectory(T=length,
                                                            sample_freq=sample_freq)
            loc = loc[:, :, :args.n_observed]
            vel = vel[:, :, :args.n_observed]
            edges = edges[:args.n_observed, :args.n_observed]

            loc_all.append(loc)
            vel_all.append(vel)
            edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
    if dataset=='charged':
        charges_all = np.stack(charges_all)
        loc_all = np.stack(loc_all)
        vel_all = np.stack(vel_all)
        edges_all = np.stack(edges_all)
        return loc_all, vel_all, edges_all, charges_all
    elif dataset=='gravity':
        force_save_all = np.stack(force_save_all)
        loc_all = np.stack(loc_all)
        vel_all = np.stack(vel_all)
        mass_all = np.stack(mass_all)
        return loc_all, vel_all, force_save_all, mass_all
    elif dataset=='springs':
        loc_all = np.stack(loc_all)
        vel_all = np.stack(vel_all)
        edges_all = np.stack(edges_all)
        return loc_all, vel_all, edges_all


if __name__ == "__main__":
    if args.simulation=='charged':
        print("Generating {} training simulations".format(args.num_train))
        loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} validation simulations".format(args.num_valid))
        loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} test simulations".format(args.num_test))
        loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                        args.length_test,
                                                                        args.sample_freq,
                                                                        args.simulation)
        path='/home/zinanzheng/project/KD/nbody/nbody/nbodydata/nbody_charged_5_4_3_200/'
        np.save(path+'loc_train' + suffix + '.npy', loc_train)
        np.save(path+'vel_train' + suffix + '.npy', vel_train)
        np.save(path+'edges_train' + suffix + '.npy', edges_train)
        np.save(path+'charges_train' + suffix + '.npy', charges_train)

        np.save(path+'loc_valid' + suffix + '.npy', loc_valid)
        np.save(path+'vel_valid' + suffix + '.npy', vel_valid)
        np.save(path+'edges_valid' + suffix + '.npy', edges_valid)
        np.save(path+'charges_valid' + suffix + '.npy', charges_valid)

        np.save(path+'loc_test' + suffix + '.npy', loc_test)
        np.save(path+'vel_test' + suffix + '.npy', vel_test)
        np.save(path+'edges_test' + suffix + '.npy', edges_test)
        np.save(path+'charges_test' + suffix + '.npy', charges_test)
    elif args.simulation=='gravity':
        print("Generating {} training simulations".format(args.num_train))
        loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} validation simulations".format(args.num_valid))
        loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} test simulations".format(args.num_test))
        loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                        args.length_test,
                                                                        args.sample_freq,
                                                                        args.simulation)
        path='/home/zinanzheng/project/KD/nbody/nbody/nbodydata/nbody_gravity_5_4_4/'
        np.save(path+'loc_train' + suffix + '.npy', loc_train)
        np.save(path+'vel_train' + suffix + '.npy', vel_train)
        np.save(path+'force_train' + suffix + '.npy', edges_train)
        np.save(path+'mass_train' + suffix + '.npy', charges_train)

        np.save(path+'loc_valid' + suffix + '.npy', loc_valid)
        np.save(path+'vel_valid' + suffix + '.npy', vel_valid)
        np.save(path+'force_valid' + suffix + '.npy', edges_valid)
        np.save(path+'mass_valid' + suffix + '.npy', charges_valid)

        np.save(path+'loc_test' + suffix + '.npy', loc_test)
        np.save(path+'vel_test' + suffix + '.npy', vel_test)
        np.save(path+'force_test' + suffix + '.npy', edges_test)
        np.save(path+'mass_test' + suffix + '.npy', charges_test)
    elif args.simulation=='springs':
        print("Generating {} training simulations".format(args.num_train))
        loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} validation simulations".format(args.num_valid))
        loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                                            args.length,
                                                                            args.sample_freq,
                                                                            args.simulation)

        print("Generating {} test simulations".format(args.num_test))
        loc_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                                        args.length_test,
                                                                        args.sample_freq,
                                                                        args.simulation)
        path='/home/zinanzheng/project/KD/nbody/nbody/nbodydata/nbody_charged_5_4_4/'
        np.save(path+'loc_train' + suffix + '.npy', loc_train)
        np.save(path+'vel_train' + suffix + '.npy', vel_train)
        np.save(path+'edges_train' + suffix + '.npy', edges_train)

        np.save(path+'loc_valid' + suffix + '.npy', loc_valid)
        np.save(path+'vel_valid' + suffix + '.npy', vel_valid)
        np.save(path+'edges_valid' + suffix + '.npy', edges_valid)

        np.save(path+'loc_test' + suffix + '.npy', loc_test)
        np.save(path+'vel_test' + suffix + '.npy', vel_test)
        np.save(path+'edges_test' + suffix + '.npy', edges_test)