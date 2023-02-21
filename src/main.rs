#![allow(dead_code)]

pub mod search;
pub mod qr;
pub mod clifford;

use std::{path::{Path, PathBuf}, sync::{Arc, mpsc, atomic}, fs::File, time::SystemTime, io::Write};
use ndarray as nd;
use ndarray_linalg::{c64, Norm};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use clap::{ValueEnum, Parser};

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
enum EmitType {
    NPY,
    QASM,
    TIKZ
}

#[derive(Debug, Parser)]
#[clap(author, version, about)]
struct Args {
    /// Number of terms to look for in the decomposition.
    #[clap(value_parser = parse_chi)]
    chi: usize,

    /// File containing the target state.
    /// This should be a numpy .npy file containing a 2^n x 2^m matrix of complex values.
    /// You can use the --tensor option to consider this state tensored with itself.
    #[clap(required=true, value_parser = parse_target)]
    target: (usize, String, nd::Array1<c64>),

    /// Output folder path.
    /// By default, this is just the current directory. Files will be output into this folder
    /// with names in the format "<TARGET>_<TENSOR>_<CHI>_<TIMESTAMP>(_<TERM>).cliffs.<EXT>" 
    /// where the timestamp is a unix timestamp in milliseconds. See '--emit' for the details.
    #[clap(long, short, value_parser = parse_output, default_value = ".")]
    output: PathBuf,

    /// Number of search attempts before stopping.
    #[clap(long, short, default_value_t = 100)]
    attempts: usize,

    /// Which outputs to produce:
    /// 'npy' - A `.npy` file containing the statevector of each term.
    /// 'qasm' - A `.qasm` file per term containing a Clifford circuit that perpares this state.
    /// 'tikz' - A `.tikz` file per term containing a drawing of the corresponding APNF ZX-diagram.
    /// The 'qasm' output is widely compatible e.g with PyZX or Qiskit, the 'tikz' outputs can be
    /// edited with TikZit or loaded directly as ZX-diagrams with PyZX. The 'npy' output can be read
    /// in Python using `numpy.load`.
    #[clap(long, short, default_value = "npy,qasm,tikz", use_value_delimiter=true)]
    emit: Vec<EmitType>,

    /// Whether to look for multiple decompositions.
    /// If set, the program will keep looking for decompositions after one is found.
    /// Only decompositions that are unique up to permutations and scalar factors will be saved. 
    #[clap(long, short)]
    multiple: bool,

    /// Number of searches to run in parallel. 
    /// Defaults to the number of threads on your machine.
    #[clap(long, short)]
    jobs: Option<usize>,

    /// Copies of the target state to tensor together.
    #[clap(long, short, default_value_t=1)]
    tensor: usize,

    /// Number of steps in the annealing schedule.
    #[clap(long, short, default_value_t=1000)]
    steps: usize,

    /// Fidelity threshold for accepting approximate decompositions.
    #[clap(long, short = 'A')]
    approx: Option<f64>,

    /// Random moves per step for simulated annealing.
    #[clap(long, default_value_t=100)]
    anneal_moves: usize,

    /// Starting temperature for simulated annealing.
    #[clap(long, default_value_t=3000.0)]
    anneal_start: f64,

    /// End temperature for simulated annealing.
    #[clap(long, default_value_t=1.0)]
    anneal_end: f64,

    /// Seed value for all randomness in the program.
    #[clap(long)]
    seed: Option<usize>
}

fn parse_chi(s: &str) -> Result<usize, String> {
    let n = s.parse::<usize>()
        .map_err(|e| format!("{}.", e))?;

    if n == 0 {
        Err("chi must be at least one.".into())
    } else {
        Ok(n)
    }
}

fn parse_target(s: &str) -> Result<(usize, String, nd::Array1<c64>), String> {
    // We should be able to open the file
    let file = File::open(s)
        .map_err(|e| e.to_string())?;

    // The file should be a valid .npy file
    let mut array = if let Ok(array) = nd::ArrayD::<c64>::read_npy(file) {
        array
    } else {
        let file = File::open(s)
            .map_err(|e| e.to_string())?;
        nd::ArrayD::<f64>::read_npy(file)
            .map_err(|e| e.to_string())?
            .map(|x| c64::from(x))
    };

    let shape = array.shape();
    let len = array.len();
    let norm = array.norm();

    // We only support 1D or 2D targets.
    if shape.len() > 2 || shape.len() == 0 {
        return Err("expected the target to be either one or two dimensional.".into())
    }

    // The dimensions must be powers of two.
    if !shape[0].is_power_of_two() || (shape.len() == 2 && !shape[1].is_power_of_two()) {
        return Err("the dimensions of the target must be powers of two.".into())
    }

    // If the norm is zero, this is just the zero state.
    if norm == 0.0 {
        return Err("the norm of the target must be non-zero".into())
    }

    // Normalize the target and transform it into 1D.
    array /= c64::from(norm);
    let array = array.into_shape((len,)).unwrap();

    // The filename should be sane.
    let name = Path::new(s).file_stem()
        .ok_or("this file does not have a valid name")?
        .to_str()
        .ok_or("this file has a non-unicode name")?
        .to_string();

    let n = len.next_power_of_two().trailing_zeros() as usize;

    Ok((n, name, array))
}

fn parse_output(s: &str) -> Result<PathBuf, String> {
    // The path should exist and be a directory.
    let path = PathBuf::from(s);
    let meta = path.metadata()
        .map_err(|e| e.to_string())?;
    if meta.is_dir() {
        Ok(path)
    } else {
        Err("this path is not a directory.".into())
    }
}

fn main() {
    // Parse the CLI arguments.
    let mut args = Args::parse();

    // Do the any tensor products necessary to get the final target vector.
    let mut target = args.target.2.slice(nd::s![.., nd::NewAxis]).to_owned();
    for _ in 1..args.tensor {
        target = nd::linalg::kron(&target, &args.target.2.slice(nd::s![.., nd::NewAxis]));
    }
    let len = target.len();
    let target = target.into_shape(len).unwrap();
    let n = args.target.0 * args.tensor;
    args.target = (n, args.target.1, target);

    // Now make args shared across threads.
    let args = Arc::new(args);

    // Create a threadpool and a progress bar manager.
    let pool = threadpool::ThreadPool::new(args.jobs.unwrap_or(num_cpus::get()));
    let bars = indicatif::MultiProgress::new();
    bars.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    println!("cliffs: searching for decomposition of {} x {} into {} terms", args.tensor, args.target.1, args.chi);

    // Create a "progress" bar that displays status information
    let status_bar = indicatif::ProgressBar::new(1);
    status_bar.set_style(indicatif::ProgressStyle::default_spinner()
        .template("[{elapsed} elapsed] {msg:.yellow.bold}"));
    let status_bar = bars.add(status_bar);
    status_bar.tick();

    // Channel for fitness values
    let (txf, rxf) = mpsc::channel::<f64>();
    // Channel for final states and GSLCs
    let (txr, rxr) = mpsc::channel::<nd::Array2<c64>>();
    // Shared integer representing number of decompositions found.
    let decomps = Arc::new(atomic::AtomicUsize::new(0));

    // Spawn a thread to update the status bar.
    let decompsf = decomps.clone();
    std::thread::spawn(move || {
        let mut best_fitness = 0.0;
        // For every attempt,
        while let Ok(fitness) = rxf.recv() {
            // keep track of the best fitness
            if fitness > best_fitness {
                best_fitness = fitness;
            }

            // Display the best fitness and number of decompositions
            let decomps = decompsf.load(atomic::Ordering::Relaxed);
            status_bar.set_message(format!("best fitness = {:.5}, decompositions found = {}", best_fitness, decomps));
        }
        
        // When we are done keep displaying the bar
        let decomps = decompsf.load(atomic::Ordering::Relaxed);
        status_bar.finish_with_message(format!("best fitness = {:.5}, decompositions found = {}", best_fitness, decomps));
    });

    // Spawn a thread to handle saving the output files
    let decomps_out = decomps.clone();
    let args_out = args.clone();
    let save_thread = std::thread::spawn(move || {
        let args = args_out;
        // Keep track of all previous decompositions that have been found
        let mut prevstates = Vec::<nd::Array2<c64>>::new();
        
        'outer: while let Ok(state) = rxr.recv() {
            // Check to make sure this is not a permutation of a previous decomposition
            for prev in &prevstates {
                let mut same = 0;
                for i in 0..args.chi {
                    for j in 0..args.chi {
                        let ci = state.column(i);
                        let cj = prev.column(j);
                        if ci.iter().zip(cj.iter()).all(|(x, y)| (x - y).norm() < 1e-6) {
                            same += 1;
                            break
                        }
                    }
                }

                if same == args.chi {
                    continue 'outer
                }
            }

            // If so we can add one to the number found
            let decomps_found = decomps_out.fetch_add(1, atomic::Ordering::Relaxed);
            // If we aren't looking for multiple decompositions, don't save this one
            if decomps_found >= 1 && !args.multiple {
                continue
            }

            // Create the stem of the output files
            let timestamp = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let stem = format!("{}_{}_{}_{}", args.target.1, args.tensor, args.chi, timestamp);

            if args.emit.contains(&EmitType::NPY) {
                // Save the state as a 2d numpy array to file
                let state_file = File::create(args.output
                    .join(format!("{}.cliffs.npy", stem))
                ).unwrap();
                state.write_npy(state_file).unwrap();
            }

            if args.emit.contains(&EmitType::QASM) || args.emit.contains(&EmitType::TIKZ) {
                // Save each component individually as a pyzx-compatible qasm and tikz file.
                for i in 0..state.shape()[1] {
                    let col = state.column(i);
                    let (graph, circ) = clifford::find_clifford(&col, 1e-6);

                    if args.emit.contains(&EmitType::TIKZ) {
                        let mut graph_file = File::create(args.output
                            .join(format!("{}_{}.cliffs.tikz", stem, i))
                        ).unwrap();
                        write!(&mut graph_file, "{}", clifford::graph_to_tikz(&graph)).unwrap();
                    }

                    if args.emit.contains(&EmitType::QASM) {
                        let mut qasm_file = File::create(args.output
                            .join(format!("{}_{}.cliffs.qasm", stem, i))
                        ).unwrap();
                        write!(&mut qasm_file, "{}", circ.to_qasm()).unwrap();
                    }
                }
            }

            prevstates.push(state);
        }
    });
   
    // Spawn a thread on the pool for each attempt
    for attempt in 0..args.attempts {
        // Create a progress bar to track the state of this attempt
        let bar = indicatif::ProgressBar::new(args.steps as u64);
        bar.set_style(indicatif::ProgressStyle::default_bar()
            .template(&format!("[{:>3}/{}] {{bar:40.cyan/blue}} {{pos:>5}}/{{len}}, {{eta:.bold}} remaining, {{msg:.bold}}", attempt+1, args.attempts))
            .progress_chars("##-"));
        let bar = bars.add(bar);
        // Clone the shared info and create a channel sender for fitness and state
        let args = args.clone();
        let txf = txf.clone();
        let txr = txr.clone();
        let decomps = decomps.clone();

        pool.execute(move || {
            bar.reset_eta();
            bar.reset_elapsed();

            // If we already finished, just exit
            if decomps.load(atomic::Ordering::Relaxed) > 0 && !args.multiple {
                bar.finish_and_clear();
                return
            }

            // Setup the random walk
            let beta = search::GeometricSequence::new(
                3000.0 / args.anneal_start,
                3000.0 / args.anneal_end,
                args.steps
            );
            let mut rw = search::RandomWalk::new(
                args.target.0,
                args.chi,
                args.anneal_moves,
                beta.clone(),
                args.target.2.clone()
            );

            bar.tick();
    
            // For each step of the walk:
            for fitness in &mut rw {
                // Send the fitness to the status bar
                bar.inc(1);
                let msg = format!("fitness = {:.3}", fitness);
                bar.set_message(msg);
                txf.send(fitness).unwrap();

                // If we are done stop here
                if decomps.load(atomic::Ordering::Relaxed) > 0 && !args.multiple {
                    bar.finish_and_clear();
                    return
                }
            }

            let (fitness, state) = rw.finish();
            txf.send(fitness).unwrap();
            let thresh = args.approx.map(|f| 1.0 - f).unwrap_or(1e-6);
            if (fitness - 1.0).abs() <= thresh {
                txr.send(state).unwrap();
            }
            bar.finish_and_clear();
        })

    }

    std::mem::drop(txf);
    std::mem::drop(txr);
    bars.join().unwrap();
    match save_thread.join() {
        Ok(_) => (),
        Err(e) => panic!("{:?} {:?}", e.downcast_ref::<&str>(), e.downcast_ref::<String>())
    }
}
