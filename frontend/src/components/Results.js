import React, { useState, useEffect } from 'react';

const API_URL = "http://localhost:8888";

const SiftVisualization = ({ members, modelData }) => {
    if (!modelData.sift_visualizations || members.length < 2) return null;
    const siftPairs = [];
    for (let j = 0; j < members.length; j++) { for (let k = j + 1; k < members.length; k++) {
        const pairKey = [members[j], members[k]].sort().join('_vs_');
        if (modelData.sift_visualizations[pairKey]) {
            siftPairs.push({ pairKey, p1: members[j], p2: members[k], img: modelData.sift_visualizations[pairKey] });
        }
    }}
    if (siftPairs.length === 0) return null;
    return (
        <div className="w-full md:w-1/2 md:pl-4 mt-4 md:mt-0">
            <div className="bg-gray-50 p-3 rounded-lg h-full"><h4 className="font-semibold text-gray-700 mb-2 text-center">ผลการวิเคราะห์ SIFT (ตัวช่วยตัดสิน):</h4>
                {siftPairs.map(({ pairKey, p1, p2, img }) => (
                    <div key={pairKey} className="mt-4 text-center"><p className="text-sm font-medium">{p1} vs {p2}</p>
                        <div className="h-40 w-full flex justify-center items-center mt-1">
                            <img src={img} alt={`SIFT for ${pairKey}`} className="max-h-full max-w-full w-auto h-auto rounded-md border shadow-sm" />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const PairwiseTable = ({ members, modelData }) => {
    if (members.length <= 1) return null;
    const allPairwiseData = modelData.face_pairwise_results || modelData.background_pairwise_results;
    if (!allPairwiseData) return null;
    const pairLookup = new Map(allPairwiseData.map(p => [p.pair.slice().sort().join('_vs_'), p]));
    const pairs = [];
    for (let j = 0; j < members.length; j++) { for (let k = j + 1; k < members.length; k++) {
        const pairKey = [members[j], members[k]].sort().join('_vs_');
        if(pairLookup.has(pairKey)) { pairs.push(pairLookup.get(pairKey)); }
    }}
    if(pairs.length === 0) return null;
    return (
        <div className="mt-6 border-t pt-4">
            <h4 className="font-semibold text-gray-700 mb-2">ผลการเปรียบเทียบรายคู่ภายในกลุ่ม:</h4>
            <div className="overflow-x-auto text-xs"><table className="min-w-full">
                <thead className="bg-gray-100"><tr>
                    <th className="p-2 text-left">ภาพที่ 1</th><th className="p-2 text-left">ภาพที่ 2</th>
                    <th className="p-2 text-right">Cosine</th><th className="p-2 text-right">Pearson</th>
                </tr></thead>
                <tbody>
                    {pairs.map(pairData => (
                         <tr key={pairData.pair.join('-')} className="border-t">
                            <td className="p-2 truncate" title={pairData.pair[0]}>{pairData.pair[0]}</td>
                            <td className="p-2 truncate" title={pairData.pair[1]}>{pairData.pair[1]}</td>
                            <td className="p-2 text-right font-medium text-blue-600">{(pairData.cosine * 100).toFixed(2)}%</td>
                            <td className="p-2 text-right font-medium text-green-600">{(pairData.pearson * 100).toFixed(2)}%</td>
                        </tr>
                    ))}
                </tbody>
            </table></div>
        </div>
    );
};

const DetailedSummaryTable = ({ groups, avgScores, dynamicTitle }) => {
    if (!groups || !avgScores) { 
        console.warn("DetailedSummaryTable: Missing groups or avgScores", { groups, avgScores });
        return <div className="text-center text-gray-500">ข้อมูลสรุปไม่พร้อมใช้งาน</div>;
    }
    const leftoverNames = ["บุคคลที่ไม่ถูกจัดกลุ่ม", "สถานที่ที่ไม่ถูกจัดกลุ่ม"];
    let scoreCounter = 0;
    return (
        <div className="bg-white p-4 rounded-xl shadow-md mb-8">
            <h3 className="text-xl font-bold text-center mb-4">{dynamicTitle}</h3>
            <div className="overflow-x-auto"><table className="min-w-full">
                <thead className="bg-gray-50"><tr>
                    <th className="p-2 text-left">ประเภทกลุ่มหลัก</th><th className="p-2">กลุ่มที่</th>
                    <th className="p-2">จำนวนภาพ</th><th className="p-2 text-left">รายชื่อไฟล์</th>
                    <th className="p-2">เฉลี่ย Cosine</th><th className="p-2">เฉลี่ย Pearson</th>
                </tr></thead>
                <tbody>
                {Object.entries(groups).flatMap(([groupName, groupList]) =>
                    groupList.map((members, i) => {
                        const scoreKey = `${groupName}_${scoreCounter}`;
                        const scores = (avgScores && avgScores[scoreKey]) ? avgScores[scoreKey] : { cosine: 0, pearson: 0 };
                        scoreCounter++;
                        return (
                            <tr key={scoreKey} className="border-t">
                                <td className="p-2">{groupName}</td>
                                <td className="p-2 text-center">{leftoverNames.includes(groupName) ? '-' : i + 1}</td>
                                <td className="p-2 text-center">{members.length}</td>
                                <td className="p-2 text-xs">{members.join(', ')}</td>

                                {members.length > 1 ? (
                                    <>
                                        <td className="p-2 text-center font-medium text-blue-600">{(scores.cosine * 100).toFixed(2)}%</td>
                                        <td className="p-2 text-center font-medium text-green-600">{(scores.pearson * 100).toFixed(2)}%</td>
                                    </>
                                ) : (
                                    <>
                                        <td className="p-2 text-center font-medium text-gray-500">-</td>
                                        <td className="p-2 text-center font-medium text-gray-500">-</td>
                                    </>
                                )}
                            </tr>
                        );
                    })
                )}
                </tbody>
            </table></div>
        </div>
    );
};

const VisualGroups = ({ title, groups, imageUrls, imageKey, avgScores, modelData }) => {
     if (!groups || !avgScores) {
         console.warn("VisualGroups: Missing groups or avgScores", { groups, avgScores });
        return <div className="text-center text-gray-500 my-6">ข้อมูลกลุ่มภาพไม่พร้อมใช้งาน</div>;
    }
    const leftoverNames = ["บุคคลที่ไม่ถูกจัดกลุ่ม", "สถานที่ที่ไม่ถูกจัดกลุ่ม"];
    const groupStyles = {"คนเดียวกัน": "green", "บุคคลที่ไม่ถูกจัดกลุ่ม": "red", "สถานที่เดียวกัน": "green", "ไม่สามารถระบุได้": "yellow", "สถานที่ที่ไม่ถูกจัดกลุ่ม": "red"};
    let scoreCounter = 0;
    return (
        <div>
            <h2 className="text-2xl font-bold my-6">{title}</h2>
            {Object.entries(groups).map(([groupName, groupList]) =>
                groupList.map((members, i) => {
                    const scoreKey = `${groupName}_${scoreCounter}`;
                    const scores = (avgScores && avgScores[scoreKey]) ? avgScores[scoreKey] : { cosine: 0, pearson: 0 };
                    const color = groupStyles[groupName] || 'gray';
                    scoreCounter++;
                    const hasSift = groupName === "ไม่สามารถระบุได้" && modelData?.sift_visualizations;
                    const borderColorClass = {
                        green: 'border-green-500', yellow: 'border-yellow-500',
                        red: 'border-red-500', gray: 'border-gray-300'
                    }[color];
                    const labelColorClasses = {
                        green: 'text-green-800 bg-green-200', yellow: 'text-yellow-800 bg-yellow-200',
                        red: 'text-red-800 bg-red-200', gray: 'text-gray-800 bg-gray-200'
                    };

                    return (
                        <div key={scoreKey} className={`bg-white p-5 rounded-xl shadow-md mb-6 border-l-4 ${borderColorClass}`}>
                            <div className="flex items-center space-x-4 mb-4">
                                <span className={`px-4 py-1 text-sm font-bold rounded-full ${labelColorClasses[color]}`}>{groupName}</span>
                                {!leftoverNames.includes(groupName) && <span className="font-semibold">กลุ่มที่ {i + 1}</span>}
                            </div>
                            <div className="border-t pt-4">
                                <div className="flex flex-wrap items-center">
                                    <p className="text-sm text-gray-600 mr-4">จำนวน {members.length} ภาพ</p>
                                    
                                    {members.length > 1 && (
                                        <div className="text-sm text-gray-600">
                                            <span>ค่าเฉลี่ยความคล้าย:</span>
                                            <span className="font-medium text-blue-600 ml-2">Cosine: {(scores.cosine * 100).toFixed(2)}%</span>
                                            <span className="font-medium text-green-600 ml-2">Pearson: {(scores.pearson * 100).toFixed(2)}%</span>
                                        </div>
                                    )}

                                </div>
                                <div className="flex flex-col md:flex-row">
                                    <div className={`w-full ${hasSift ? 'md:w-1/2' : 'w-full'}`}>
                                        <div className="mt-4 grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
                                            {members.map(fname => {
                                                const urls = imageUrls?.[fname];
                                                const imgSrc = urls?.[imageKey] || urls?.original;
                                                const dateTime = urls?.metadata_datetime || 'N/A';

                                                return imgSrc ? (
                                                    <div key={fname} className="text-center">
                                                        <div className="relative h-40 w-full flex justify-center items-center bg-gray-100 border rounded-md">
                                                            <img
                                                                src={imgSrc}
                                                                alt={fname}
                                                                className="max-h-full max-w-full w-auto h-auto object-contain rounded-md shadow-sm"
                                                            />
                                                            <div className="absolute bottom-1 left-1 bg-black bg-opacity-60 text-white text-xs px-1 py-0.5 rounded font-mono">
                                                                {dateTime}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs mt-1 text-gray-600 truncate" title={fname}>{fname}</p>
                                                    </div>
                                                ) : (
                                                     <div key={fname} className="text-center">
                                                         <div className="h-40 w-full flex justify-center items-center bg-gray-100 border rounded-md text-gray-400 text-xs">
                                                            No Image URL
                                                         </div>
                                                         <p className="text-xs mt-1 text-gray-600 truncate" title={fname}>{fname}</p>
                                                     </div>
                                                );
                                            })}
                                        </div>
                                        {groupName !== "ไม่สามารถระบุได้" && <PairwiseTable members={members} modelData={modelData} />}
                                    </div>
                                    {hasSift && <SiftVisualization members={members} modelData={modelData} />}
                                </div>
                            </div>
                        </div>
                    );
                })
            )}
        </div>
    );
};


const Results = ({ results, jobId }) => {
    const [activeTab, setActiveTab] = useState('');

    const models = results?.models || {}; 
    const imageUrls = results?.all_images_data_urls || {}; 

    const modelNames = React.useMemo(() => {
        return models ? Object.keys(models) : [];
    }, [models]);

    useEffect(() => {
        if (modelNames.length > 0 && activeTab === '') {
            setActiveTab(modelNames[0]);
        }
    }, [modelNames, activeTab]);

    const firstModelData = modelNames.length > 0 ? models[modelNames[0]] : null;
    const imageCount = firstModelData ? firstModelData.image_count : 'N/A';
    const processingTime = new Date().toLocaleString('en-GB', { 
        day: '2-digit', 
        month: '2-digit', 
        year: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    }).replace(',', ''); 


    if (!results || typeof results !== 'object' || modelNames.length === 0) {
        console.error("Invalid or empty results object received:", results);
        return <div className="text-center p-8">ไม่พบผลการวิเคราะห์สำหรับโมเดลใดๆ</div>;
    }


    const MasterSummaryTable = () => (
        <div className="bg-white p-6 rounded-xl shadow-lg mb-8">
            <h2 className="text-2xl font-bold text-center mb-6">ตารางสรุปเปรียบเทียบประสิทธิภาพโมเดล</h2>
            <div className="overflow-x-auto"><table className="min-w-full">
                <thead className="bg-gray-50"><tr>
                    <th className="p-3 font-semibold text-left">ชื่อโมเดล</th><th className="p-3 font-semibold text-left">โมเดลเสริม</th>
                    <th className="p-3 font-semibold">ประเภท</th><th className="p-3 font-semibold">จำนวนภาพ</th>
                    <th className="p-3 font-semibold">เวลาประมวลผล (วินาที)</th>
                </tr></thead>
                <tbody>
                {modelNames.map(modelName => {
                    const modelData = models[modelName];
                    if (typeof modelData !== 'object' || modelData === null || !('processing_time' in modelData)) {
                         console.warn(`Skipping invalid model data for ${modelName} in MasterSummaryTable`);
                         return null;
                    }
                    const isFaceModel = 'face_groups' in modelData;
                    const modelType = isFaceModel ? 'ใบหน้า' : 'ฉากหลัง';
                    const auxModel = isFaceModel ? 'retina (face detection)' : 'deeplab (background segmentation)';
                    return (
                        <tr key={modelName} className="border-t">
                            <td className="p-3 text-left font-semibold">{modelName}</td>
                            <td className="p-3 text-left text-gray-600">{auxModel}</td>
                            <td className="p-3 text-center">{modelType}</td>
                            <td className="p-3 text-center">{modelData.image_count ?? 'N/A'}</td>
                            <td className="p-3 text-center">{modelData.processing_time.toFixed(2)}</td>
                        </tr>
                    );
                })}
                </tbody>
            </table></div>
        </div>
    );

    return (
        <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">

            <div className="bg-white p-4 rounded-xl shadow-lg mb-8 text-center">
                <h2 className="text-xl font-bold text-gray-800 mb-4">ข้อมูลทั่วไป</h2>
                <div className="grid grid-cols-2 gap-4 max-w-sm mx-auto text-sm">
                    <div className="font-semibold text-gray-700 text-right">วัน-เวลาประมวลผล:</div>
                    <div className="text-gray-900 text-left font-semibold">{processingTime}</div>
                    <div className="font-semibold text-gray-700 text-right">จำนวนภาพ:</div>
                    <div className="text-gray-900 text-left font-semibold">{imageCount}</div>
                </div>
            </div>

            <MasterSummaryTable />

            <div className="bg-white p-6 rounded-xl shadow-lg">
                <h2 className="text-2xl font-bold text-center mb-6">ผลการวิเคราะห์รายโมเดล</h2>
                <div className="border-b border-gray-200">
                    <div className="flex flex-wrap -mb-px">
                        {modelNames.map(name => {
                            const modelData = models[name];
                            if (typeof modelData !== 'object' || modelData === null) return null;
                            const isFaceModel = 'face_groups' in modelData;
                            const modelTypeColor = isFaceModel ? 'text-blue-700' : 'text-green-700';
                            const isActive = activeTab === name;

                            return (
                                <button key={name}
                                    className={`text-lg py-3 px-4 border-b-2 font-medium transition-colors duration-200 ${modelTypeColor} ${
                                        isActive
                                        ? 'border-blue-500'
                                        : 'border-transparent hover:border-gray-300'
                                    }`}
                                    onClick={() => setActiveTab(name)}>
                                    {name}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {modelNames.map(modelName => {
                    const modelData = models[modelName];
                    if (typeof modelData !== 'object' || modelData === null) return null;

                    const isFaceModel = 'face_groups' in modelData;
                    const tableTitle = isFaceModel ? 'สรุปผลการจัดกลุ่มใบหน้า' : 'สรุปผลการจัดกลุ่มฉากหลัง';
                    const visualTitle = isFaceModel ? 'กลุ่มใบหน้า' : 'กลุ่มฉากหลัง';
                    const groupsData = isFaceModel ? modelData.face_groups : modelData.background_groups;
                    const avgScoresData = isFaceModel ? modelData.avg_face_scores : modelData.avg_background_scores;
                    const imageKey = isFaceModel ? 'face' : 'background';

                    const validGroups = groupsData && typeof groupsData === 'object' ? groupsData : {};
                    const validAvgScores = avgScoresData && typeof avgScoresData === 'object' ? avgScoresData : {};


                    return (
                        <div key={modelName} style={{ display: activeTab === modelName ? 'block' : 'none' }}>
                            <DetailedSummaryTable 
                                groups={validGroups} 
                                avgScores={validAvgScores} 
                                dynamicTitle={tableTitle} 
                            />
                            <VisualGroups
                                 title={visualTitle}
                                 groups={validGroups}
                                 imageUrls={imageUrls} 
                                 imageKey={imageKey}
                                 avgScores={validAvgScores}
                                 modelData={modelData}
                            />
                        </div>
                    )
                })}
            </div>

            <div className="text-center mt-12 space-x-4">
                <a href={`${API_URL}/api/download_report/${jobId}`} className="bg-purple-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-purple-700 transition">ดาวน์โหลด PDF</a>
                <a href="/" className="bg-gray-500 text-white font-bold py-3 px-8 rounded-lg hover:bg-gray-600 transition">วิเคราะห์ใหม่อีกครั้ง</a>
            </div>
        </div>
    );
};

export default Results;