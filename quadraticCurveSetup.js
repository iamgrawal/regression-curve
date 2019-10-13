// P5.js variables
let x_vals = [];
let y_vals = [];
let dragging = false;

// Defining Variables for the polynomial equation y = ax^2 + bx + c
let a, b, c;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function setup() {
	createCanvas(400, 400);
	a = tf.variable(tf.scalar(random(-1, 1)));
	b = tf.variable(tf.scalar(random(-1, 1)));
	c = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
	return pred
		.sub(labels)
		.square()
		.mean();
}

function predict(x) {
	const xs = tf.tensor1d(x);
	// y = ax^2 + bx + c
	const ys = xs
		.square()
		.mul(a)
		.add(xs.mul(b))
		.add(c);
	return ys;
}

// Mouse actions functions
function mousePressed() {
	dragging = true;
}

function mouseReleased() {
	dragging = false;
}

function draw() {
	if (dragging) {
		let x = map(mouseX, 0, width, -1, 1);
		let y = map(mouseY, 0, height, 1, -1);
		x_vals.push(x);
		y_vals.push(y);
	} else {
		tf.tidy(() => {
			if (x_vals.length > 0) {
				const ys = tf.tensor1d(y_vals);
				optimizer.minimize(() => loss(predict(x_vals), ys));
			}
		});
	}

	background(0);

	stroke(255);
	strokeWeight(8);
	for (let i = 0; i < x_vals.length; i++) {
		let px = map(x_vals[i], -1, 1, 0, width);
		let py = map(y_vals[i], -1, 1, height, 0);
		point(px, py);
	}

	const curveX = [];
	for (let x = -1; x <= 1; x += 0.05) {
		curveX.push(x);
	}

	const ys = tf.tidy(() => predict(curveX));
	let curveY = ys.dataSync();
	ys.dispose();

	beginShape();
	noFill();
	stroke(255);
	strokeWeight(2);
	for (let i = 0; i < curveX.length; i++) {
		let x = map(curveX[i], -1, 1, 0, width);
		let y = map(curveY[i], -1, 1, height, 0);
		vertex(x, y);
	}
	endShape();

	// If you want to check how much memory tensors are useing GPU Memory
	// console.log(tf.memory().numTensors);
}
