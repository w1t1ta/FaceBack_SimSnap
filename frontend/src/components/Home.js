import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API_URL = "http://localhost:8000";

const Home = () => {
    const [files, setFiles] = useState([]);
    const [mode, setMode] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isDragOver, setIsDragOver] = useState(false);
    const navigate = useNavigate();

    const handleFileChange = (event) => {
        const newFiles = Array.from(event.target.files);
        setFiles(prevFiles => {
            const existingFilenames = new Set(prevFiles.map(f => f.name));
            const uniqueNewFiles = newFiles.filter(f => !existingFilenames.has(f.name));
            return [...prevFiles, ...uniqueNewFiles];
        });
    };

    const removeFile = (indexToRemove) => {
        const updatedFiles = files.filter((_, index) => index !== indexToRemove);
        setFiles(updatedFiles);
        if (updatedFiles.length === 0) {
            setMode('');
        }
    };
    
    const handleDragEvents = (e, over) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(over);
    };
    
    const handleDrop = (e) => {
        handleDragEvents(e, false);
        const droppedFiles = Array.from(e.dataTransfer.files);
        setFiles(prevFiles => {
             const existingFilenames = new Set(prevFiles.map(f => f.name));
             const uniqueNewFiles = droppedFiles.filter(f => f.type.startsWith('image/') && !existingFilenames.has(f.name));
             return [...prevFiles, ...uniqueNewFiles];
        });
    };


    const handleSubmit = async (event) => {
        event.preventDefault();
        if (files.length < 2 || files.length > 100 || !mode) return;
        setIsLoading(true);

        const formData = new FormData();
        files.forEach(file => formData.append('images', file));

        try {
            const response = await fetch(`${API_URL}/api/submit_images`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to submit images.');
            }

            const data = await response.json();
            const jobId = data.job_id;

            if (!jobId) {
                throw new Error("Could not retrieve Job ID from the server.");
            }
            
            if (mode === 'quick_analysis') {
                navigate(`/select-model/${jobId}`);
            } else {
                navigate(`/processing/${jobId}?mode=comparison`);
            }
        } catch (error) {
            console.error('Error submitting images:', error);
            alert(`Error: ${error.message}`);
            setIsLoading(false);
        }
    };
    
    const isSubmitDisabled = files.length < 2 || files.length > 100 || isLoading || !mode;

    return (
        <div className="max-w-4xl mx-auto p-4 sm:p-8">
            <div className="bg-white p-8 rounded-xl shadow-lg text-center transition-all duration-300">
                <div className="flex justify-center">
                     <img src="/cover/cover.png" alt="Cover" className="h-24" />
                </div>
                <h1 className="text-3xl font-bold text-gray-800 mt-4">กรุณาอัปโหลดรูปภาพ</h1>
                <p className="text-gray-500 mt-2">เพื่อเปรียบเทียบประสิทธิภาพโมเดล (อย่างน้อย 2 ถึง 100 ภาพ)</p>

                <form onSubmit={handleSubmit}>
                    <div className="mt-8">
                        <label 
                            htmlFor="image-input" 
                            className={`flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-lg cursor-pointer transition-all duration-300 ${isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-400 bg-white hover:border-blue-500'}`}
                            onDragOver={(e) => handleDragEvents(e, true)}
                            onDragLeave={(e) => handleDragEvents(e, false)}
                            onDrop={handleDrop}
                        >
                            <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-4-4V7a4 4 0 014-4h10a4 4 0 014 4v5a4 4 0 01-4 4H7z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 16v1a2 2 0 01-2 2H3a2 2 0 01-2-2V7a2 2 0 012-2h2m12 14l-4-4m0 0l-4 4m4-4v12"></path></svg>
                            <p className="text-lg font-semibold text-gray-600">ลากและวางไฟล์ที่นี่ หรือ คลิกเพื่ออัปโหลด</p>
                            <p className="text-xs text-gray-400 mt-2">รองรับไฟล์ JPG, PNG (สูงสุด 100 ภาพ)</p>
                        </label>
                        <input id="image-input" name="images" type="file" className="sr-only" accept="image/png, image/jpeg" multiple onChange={handleFileChange} />
                         <p id="file-info" className={`mt-4 font-medium transition-colors duration-300 ${ (files.length > 0 && (files.length < 2 || files.length > 100)) ? 'text-red-500' : 'text-gray-600'}`}>
                            {files.length === 0 ? 'ยังไม่ได้เลือกไฟล์' : (files.length < 2 || files.length > 100) ? `ต้องเลือก 2 - 100 ภาพ! (คุณเลือก ${files.length} ภาพ)`: `เลือกแล้ว ${files.length} ภาพ`}
                        </p>
                    </div>

                    {files.length > 0 && (
                        <div className="mt-6 p-4 bg-gray-50 rounded-lg max-h-60 overflow-y-auto grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-4">
                            {files.map((file, index) => (
                                <div key={file.name + index} className="relative group aspect-square">
                                    <img src={URL.createObjectURL(file)} alt={file.name} className="w-full h-full object-cover rounded-md shadow-md" />
                                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center rounded-md">
                                        <button type="button" onClick={() => removeFile(index)} className="text-white text-2xl font-bold opacity-0 group-hover:opacity-100 transition-opacity duration-300">&times;</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="mt-8">
                        <div className="flex justify-center space-x-4 mb-6">
                            <button 
                                type="button" 
                                onClick={() => setMode('comparison')} 
                                disabled={files.length === 0}
                                className={`font-bold py-2 px-6 rounded-lg transition-all duration-300 ${mode === 'comparison' ? 'bg-blue-600 text-white shadow-md' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'} disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                                โหมดเปรียบเทียบ
                            </button>
                            <button 
                                type="button" 
                                onClick={() => setMode('quick_analysis')}
                                disabled={files.length === 0} 
                                className={`font-bold py-2 px-6 rounded-lg transition-all duration-300 ${mode === 'quick_analysis' ? 'bg-blue-600 text-white shadow-md' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'} disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                                โหมดเจาะจง
                            </button>
                        </div>
                        <button type="submit" disabled={isSubmitDisabled} className="w-full md:w-1/2 bg-green-600 text-white font-bold py-4 px-12 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl disabled:shadow-none transform hover:-translate-y-1 disabled:transform-none">
                            {isLoading ? 'กำลังอัปโหลด...' : 'เริ่มการวิเคราะห์'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Home;