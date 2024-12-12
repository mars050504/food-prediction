/** @format */

const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/inputError");

async function predictClassification(model, image) {
	try {

		const tensor            = tf.node.decodeJpeg(image).resizeNearestNeighbor([224, 224]).expandDims().toFloat();
		const prediction        = model.predict(tensor);
		const score             = await prediction.data();
		const confidenceScore   = Math.max(...score) * 100;
        
        let result = { confidenceScore, label: "Cabbage", suggestion: "Fry and cook" };
        if (confidenceScore < 1) {
            result.label        = "Non-cabbage";
            result.suggestion   = "tolong berikan foto cabbage"
        }
        
        return result;
    } catch (error) {
		throw new InputError("Terjadi kesalahan dalam melakukan prediksi");
	}
}

module.exports = predictClassification;